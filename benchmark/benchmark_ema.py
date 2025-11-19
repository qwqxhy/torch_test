import argparse
import time
from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

# Make project root importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_utilization import GpuUtilizationMonitor

from ema import torch_ema


GPU_BENCH_ITERS = 100  # number of repeated GPU runs for more stable timing/utilization


def load_prices(csv_path: Path) -> np.ndarray:
    """
    Load price matrix from CSV.
    CSV is expected to have a header row; rows = time, columns = stocks.
    Returns array of shape (T, S).
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return data


def ema_cpu_serial(prices_ts_first: np.ndarray, span: int) -> np.ndarray:
    """
    CPU baseline: vectorised implementation using torch_ema on CPU
    over all stocks at once (same math as GPU version).
    prices_ts_first: shape (T, S)  (T = time, S = stocks)
    Returns array of shape (T, S).
    """
    prices_stocks_first = prices_ts_first.T  # (S, T)
    dev = torch.device("cpu")
    price_cpu = torch.as_tensor(prices_stocks_first, dtype=torch.float32, device=dev)
    ema_cpu = torch_ema(price_cpu, span, device=dev)  # (S, T)
    return ema_cpu.detach().cpu().numpy().T


def ema_cpu_parallel(
    prices_ts_first: np.ndarray,
    span: int,
    num_workers: int | None = None,
) -> np.ndarray:
    """
    CPU multi-threaded baseline: split stocks across threads and run
    ema_cpu_serial on column chunks.
    prices_ts_first: shape (T, S)
    Returns array of shape (T, S).
    """
    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 1

    T, S = prices_ts_first.shape
    out = np.empty((T, S), dtype=np.float32)

    col_indices = np.array_split(np.arange(S), num_workers)

    def worker(cols: np.ndarray) -> None:
        if cols.size == 0:
            return
        out[:, cols] = ema_cpu_serial(prices_ts_first[:, cols], span)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(worker, col_indices))

    return out


def ema_gpu_conv_all(
    prices_ts_first: np.ndarray,
    span: int,
    device: str,
) -> tuple[np.ndarray, float, int | None, float | None]:
    """
    GPU version: all stocks in one conv1d call via torch_ema.
    prices_ts_first: shape (T, S) on CPU, float32.
    Returns array of shape (T, S) on CPU.
    """
    prices_stocks_first = prices_ts_first.T  # (S, T)

    dev = torch.device(device)
    peak_mem_bytes: int | None = None
    avg_gpu_util: float | None = None

    # Prepare device and reset CUDA memory stats if needed
    if dev.type == "cuda":
        # Some torch versions require an explicit index (e.g. cuda:0)
        index = dev.index if dev.index is not None else 0
        torch.cuda.set_device(index)
        dev = torch.device(f"cuda:{index}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        monitor = GpuUtilizationMonitor(device_index=index)
        monitor.start()
    else:
        monitor = None

    price_gpu = torch.as_tensor(prices_stocks_first, dtype=torch.float32, device=dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Run multiple times to better expose sustained GPU utilization.
    for _ in range(GPU_BENCH_ITERS):
        ema_gpu = torch_ema(price_gpu, span, device=dev)  # (S, T)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        # Peak allocated memory during this run (in bytes)
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        if monitor is not None:
            avg_gpu_util = monitor.stop()
    t1 = time.perf_counter()

    ema_cpu = ema_gpu.detach().cpu().numpy().T
    # Report average time per iteration for fair comparison with single-run CPU.
    return ema_cpu, (t1 - t0) / GPU_BENCH_ITERS, peak_mem_bytes, avg_gpu_util


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EMA: CPU serial per stock vs GPU conv over all stocks.")
    parser.add_argument("--csv", type=str, default="data/prices_3000x5000.csv", help="Input CSV path.")
    parser.add_argument("--span", type=int, default=20, help="EMA span.")
    parser.add_argument("--threads", type=int, default=None, help="Threads for CPU parallel benchmark (default: os.cpu_count()).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for GPU benchmark, e.g. cuda or cuda:0.")

    args = parser.parse_args()
    csv_path = Path(args.csv)

    prices = load_prices(csv_path)  # (T, S)
    T, S = prices.shape
    print(f"Loaded prices from {csv_path}, shape = (T={T}, S={S})")

    # CPU serial
    t0 = time.perf_counter()
    ema_cpu = ema_cpu_serial(prices, args.span)
    t1 = time.perf_counter()
    cpu_time = t1 - t0
    print(f"[CPU] EMA span={args.span}: {cpu_time:.4f} s")

    # CPU multi-threaded
    t0 = time.perf_counter()
    ema_cpu_mt = ema_cpu_parallel(prices, args.span, num_workers=args.threads)
    t1 = time.perf_counter()
    cpu_mt_time = t1 - t0
    threads_used = args.threads if args.threads and args.threads > 0 else (os.cpu_count() or 1)
    print(f"[CPU-mt] EMA span={args.span}, threads={threads_used}: {cpu_mt_time:.4f} s")

    # GPU conv (if device available)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; skipping GPU benchmark.")
        return

    ema_gpu, gpu_time, gpu_mem, gpu_util = ema_gpu_conv_all(prices, args.span, device=args.device)
    print(f"[GPU] EMA span={args.span} on {args.device}: {gpu_time:.4f} s")
    if gpu_mem is not None:
        print(f"[GPU] Peak memory allocated: {gpu_mem / (1024 ** 2):.2f} MB")
    if gpu_util is not None:
        print(f"[GPU] Average utilization: {gpu_util:.2f}%")
    else:
        print("[GPU] Average utilization: N/A (utilization metrics unavailable)")

    # Speedups
    speedup_serial_gpu = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    speedup_mt_gpu = cpu_mt_time / gpu_time if gpu_time > 0 else float("inf")
    speedup_serial_mt = cpu_time / cpu_mt_time if cpu_mt_time > 0 else float("inf")
    print(f"Speedup (CPU-serial / GPU)   = {speedup_serial_gpu:.2f}x")
    print(f"Speedup (CPU-mt / GPU)       = {speedup_mt_gpu:.2f}x")
    print(f"Speedup (CPU-serial / CPU-mt) = {speedup_serial_mt:.2f}x")

    # Optional quick correctness checks
    diff_gpu = np.nanmax(np.abs(ema_cpu - ema_gpu))
    print(f"Max abs diff between CPU-serial and GPU EMA (ignoring NaN): {diff_gpu:.6g}")

    diff_mt = np.nanmax(np.abs(ema_cpu - ema_cpu_mt))
    print(f"Max abs diff between CPU-serial and CPU-mt EMA (ignoring NaN): {diff_mt:.6g}")


if __name__ == "__main__":
    main()
