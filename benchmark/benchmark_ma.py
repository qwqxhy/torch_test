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
from ma import moving_average_matrix, torch_ma


GPU_BENCH_ITERS = 1000  # number of repeated GPU runs for more stable timing/utilization


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


def ma_cpu_serial(prices_ts_first: np.ndarray, window: int) -> np.ndarray:
    """
    CPU baseline using an O(T * S) prefix-sum implementation over all stocks.
    prices_ts_first: shape (T, S)  (T = time, S = stocks)
    Returns array of shape (T, S), same semantics as moving_average_matrix:
    first window-1 positions are NaN.
    """
    T, S = prices_ts_first.shape
    out = np.full((T, S), np.nan, dtype=np.float32)

    if window <= 0 or T < window:
        return out

    # Cumulative sum along time axis; use higher precision for accumulation
    cs = np.cumsum(prices_ts_first, axis=0, dtype=np.float64)  # (T, S)

    # For t >= window-1: sum_{k=0..window-1} price[t-k]
    # = cs[t] - cs[t-window] (with cs[-1] treated as 0)
    # Build shifted cumulative sum with a leading zero row.
    cs_shifted = np.vstack(
        [np.zeros((1, S), dtype=cs.dtype), cs[:-window, :]]
    )  # (T - window + 1, S) after slicing below

    window_sums = cs[window - 1 :, :] - cs_shifted  # (T - window + 1, S)
    out[window - 1 :, :] = (window_sums / float(window)).astype(np.float32)
    return out


def ma_cpu_parallel(
    prices_ts_first: np.ndarray,
    window: int,
    num_workers: int | None = None,
) -> np.ndarray:
    """
    CPU multi-threaded baseline: each stock (column) is processed in a thread.
    prices_ts_first: shape (T, S)
    Returns array of shape (T, S).
    """
    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 1

    T, S = prices_ts_first.shape
    out = np.empty((T, S), dtype=np.float32)

    # Split columns into roughly equal chunks for each worker
    col_indices = np.array_split(np.arange(S), num_workers)

    def worker(cols: np.ndarray) -> None:
        if cols.size == 0:
            return
        out[:, cols] = ma_cpu_serial(prices_ts_first[:, cols], window)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(worker, col_indices))

    return out


def ma_gpu_conv_all(
    prices_ts_first: np.ndarray,
    window: int,
    device: str,
) -> tuple[np.ndarray, float, int | None, float | None]:
    """
    GPU version: all stocks in one conv1d call.
    prices_ts_first: shape (T, S) on CPU, float32.
    Returns array of shape (T, S) on CPU.
    """
    # torch_ma expects (S, T)
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

        # Start GPU utilization monitoring on the chosen device.
        monitor = GpuUtilizationMonitor(device_index=index)
        monitor.start()
    else:
        monitor = None

    # Move once to target device and run conv
    price_gpu = torch.as_tensor(prices_stocks_first, dtype=torch.float32, device=dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Run multiple times to better expose sustained GPU utilization.
    for _ in range(GPU_BENCH_ITERS):
        ma_gpu = torch_ma(price_gpu, window, device=dev)  # (S, T)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        # Peak allocated memory during this run (in bytes)
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        if monitor is not None:
            avg_gpu_util = monitor.stop()
    t1 = time.perf_counter()
    print(t1 - t0)

    # Bring back to CPU and restore (T, S) layout
    ma_cpu = ma_gpu.detach().cpu().numpy().T
    # Report average time per iteration for fair comparison with single-run CPU.
    return ma_cpu, (t1 - t0) / GPU_BENCH_ITERS, peak_mem_bytes, avg_gpu_util


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MA: CPU serial per stock vs GPU conv over all stocks.")
    parser.add_argument("--csv", type=str, default="data/prices_3000x5000.csv", help="Input CSV path.")
    parser.add_argument("--window", type=int, default=20, help="MA window length.")
    parser.add_argument("--threads", type=int, default=None, help="Threads for CPU parallel benchmark (default: os.cpu_count()).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for GPU benchmark, e.g. cuda or cuda:0.")

    args = parser.parse_args()
    csv_path = Path(args.csv)

    prices = load_prices(csv_path)  # (T, S)
    T, S = prices.shape
    print(f"Loaded prices from {csv_path}, shape = (T={T}, S={S})")

    # CPU serial
    t0 = time.perf_counter()
    ma_cpu = ma_cpu_serial(prices, args.window)
    t1 = time.perf_counter()
    cpu_time = t1 - t0
    print(f"[CPU] MA window={args.window}: {cpu_time:.4f} s")

    # CPU multi-threaded
    t0 = time.perf_counter()
    ma_cpu_mt = ma_cpu_parallel(prices, args.window, num_workers=args.threads)
    t1 = time.perf_counter()
    cpu_mt_time = t1 - t0
    threads_used = args.threads if args.threads and args.threads > 0 else (os.cpu_count() or 1)
    print(f"[CPU-mt] MA window={args.window}, threads={threads_used}: {cpu_mt_time:.4f} s")

    # GPU conv (if device available)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; skipping GPU benchmark.")
        return

    ma_gpu, gpu_time, gpu_mem, gpu_util = ma_gpu_conv_all(prices, args.window, device=args.device)
    print(f"[GPU] MA window={args.window} on {args.device}: {gpu_time:.4f} s")
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
    print(f"Speedup (CPU-serial / GPU) = {speedup_serial_gpu:.2f}x")
    print(f"Speedup (CPU-mt / GPU)      = {speedup_mt_gpu:.2f}x")
    print(f"Speedup (CPU-serial / CPU-mt) = {speedup_serial_mt:.2f}x")

    # Optional quick correctness checks
    diff_gpu = np.nanmax(np.abs(ma_cpu - ma_gpu))
    print(f"Max abs diff between CPU-serial and GPU MA (ignoring NaN): {diff_gpu:.6g}")

    diff_mt = np.nanmax(np.abs(ma_cpu - ma_cpu_mt))
    print(f"Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): {diff_mt:.6g}")


if __name__ == "__main__":
    main()
