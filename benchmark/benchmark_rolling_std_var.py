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

from ma import torch_ma


GPU_BENCH_ITERS = 10  # number of repeated GPU runs for more stable timing/utilization


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


def rolling_std_var_cpu(
    prices_ts_first: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CPU baseline: rolling variance / std using prefix sums over all stocks.

    prices_ts_first: shape (T, S)  (T = time, S = stocks)
    window: rolling window length

    Returns (std, var), both shape (T, S), with the first window-1 entries NaN.
    Variance is the population variance over the window:
        var = E[x^2] - (E[x])^2
    """
    T, S = prices_ts_first.shape
    std = np.full((T, S), np.nan, dtype=np.float32)
    var = np.full((T, S), np.nan, dtype=np.float32)

    if window <= 0 or T < window:
        return std, var

    # Use higher precision for accumulation
    x = prices_ts_first.astype(np.float64, copy=False)
    cs = np.cumsum(x, axis=0, dtype=np.float64)
    cs2 = np.cumsum(x * x, axis=0, dtype=np.float64)

    # Build shifted cumulative sums with leading zeros for windowed diffs
    zeros = np.zeros((1, S), dtype=np.float64)
    cs_shifted = np.vstack([zeros, cs[:-window, :]])
    cs2_shifted = np.vstack([zeros, cs2[:-window, :]])

    sum_x = cs[window - 1 :, :] - cs_shifted
    sum_x2 = cs2[window - 1 :, :] - cs2_shifted

    mean = sum_x / float(window)
    ex2 = sum_x2 / float(window)
    var_win = ex2 - mean * mean
    # Avoid tiny negative due to numerical error
    var_win = np.maximum(var_win, 0.0)

    var[window - 1 :, :] = var_win.astype(np.float32)
    std[window - 1 :, :] = np.sqrt(var_win).astype(np.float32)
    return std, var


def rolling_std_var_cpu_parallel(
    prices_ts_first: np.ndarray,
    window: int,
    num_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CPU multi-threaded baseline: split stocks across threads.

    prices_ts_first: shape (T, S)
    Returns (std, var), both shape (T, S).
    """
    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 1

    T, S = prices_ts_first.shape
    std = np.empty((T, S), dtype=np.float32)
    var = np.empty((T, S), dtype=np.float32)

    col_indices = np.array_split(np.arange(S), num_workers)

    def worker(cols: np.ndarray) -> None:
        if cols.size == 0:
            return
        std_chunk, var_chunk = rolling_std_var_cpu(prices_ts_first[:, cols], window)
        std[:, cols] = std_chunk
        var[:, cols] = var_chunk

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(worker, col_indices))

    return std, var


def rolling_std_var_gpu(
    prices_ts_first: np.ndarray,
    window: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, int | None, float | None]:
    """
    GPU version: reuse torch_ma on price and price^2 to get mean and E[x^2].

    prices_ts_first: shape (T, S) on CPU, float32.
    Returns (std_cpu, var_cpu, elapsed_time, peak_mem_bytes).
    """
    prices_stocks_first = prices_ts_first.T  # (S, T)

    dev = torch.device(device)
    peak_mem_bytes: int | None = None
    avg_gpu_util: float | None = None

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
        mean = torch_ma(price_gpu, window, device=dev)  # (S, T)
        mean_sq = torch_ma(price_gpu * price_gpu, window, device=dev)  # (S, T)
    var_gpu = torch.clamp(mean_sq - mean * mean, min=0.0)
    std_gpu = torch.sqrt(var_gpu)

    if dev.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        if monitor is not None:
            avg_gpu_util = monitor.stop()
    t1 = time.perf_counter()

    std_cpu = std_gpu.detach().cpu().numpy().T
    var_cpu = var_gpu.detach().cpu().numpy().T
    # Report average time per iteration for fair comparison with single-run CPU.
    return std_cpu, var_cpu, (t1 - t0) / GPU_BENCH_ITERS, peak_mem_bytes, avg_gpu_util


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rolling Std/Variance: CPU serial/mt vs GPU conv over all stocks."
    )
    parser.add_argument("--csv", type=str, default="data/prices_3000x5000.csv", help="Input CSV path.")
    parser.add_argument("--window", type=int, default=20, help="Rolling window length.")
    parser.add_argument("--threads", type=int, default=None, help="Threads for CPU parallel benchmark (default: os.cpu_count()).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for GPU benchmark, e.g. cuda or cuda:0.")

    args = parser.parse_args()
    csv_path = Path(args.csv)

    prices = load_prices(csv_path)  # (T, S)
    T, S = prices.shape
    print(f"Loaded prices from {csv_path}, shape = (T={T}, S={S})")

    # CPU serial
    t0 = time.perf_counter()
    std_cpu, var_cpu = rolling_std_var_cpu(prices, args.window)
    t1 = time.perf_counter()
    cpu_time = t1 - t0
    print(f"[CPU] rolling window={args.window}: {cpu_time:.4f} s")

    # CPU multi-threaded
    t0 = time.perf_counter()
    std_cpu_mt, var_cpu_mt = rolling_std_var_cpu_parallel(prices, args.window, num_workers=args.threads)
    t1 = time.perf_counter()
    cpu_mt_time = t1 - t0
    threads_used = args.threads if args.threads and args.threads > 0 else (os.cpu_count() or 1)
    print(f"[CPU-mt] rolling window={args.window}, threads={threads_used}: {cpu_mt_time:.4f} s")

    # GPU conv (if device available)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; skipping GPU benchmark.")
        return

    std_gpu, var_gpu, gpu_time, gpu_mem, gpu_util = rolling_std_var_gpu(prices, args.window, device=args.device)
    print(f"[GPU] rolling window={args.window} on {args.device}: {gpu_time:.4f} s")
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
    print(f"Speedup (CPU-serial / GPU)    = {speedup_serial_gpu:.2f}x")
    print(f"Speedup (CPU-mt / GPU)        = {speedup_mt_gpu:.2f}x")
    print(f"Speedup (CPU-serial / CPU-mt) = {speedup_serial_mt:.2f}x")

    # Optional quick correctness checks (ignore NaNs)
    mask = ~np.isnan(std_cpu)
    max_diff_std_gpu = np.max(np.abs(std_cpu[mask] - std_gpu[mask]))
    max_diff_std_mt = np.max(np.abs(std_cpu[mask] - std_cpu_mt[mask]))
    print(f"Max abs diff Std (CPU-serial vs GPU, ignoring NaN): {max_diff_std_gpu:.6g}")
    print(f"Max abs diff Std (CPU-serial vs CPU-mt, ignoring NaN): {max_diff_std_mt:.6g}")

    mask_var = ~np.isnan(var_cpu)
    max_diff_var_gpu = np.max(np.abs(var_cpu[mask_var] - var_gpu[mask_var]))
    max_diff_var_mt = np.max(np.abs(var_cpu[mask_var] - var_cpu_mt[mask_var]))
    print(f"Max abs diff Var (CPU-serial vs GPU, ignoring NaN): {max_diff_var_gpu:.6g}")
    print(f"Max abs diff Var (CPU-serial vs CPU-mt, ignoring NaN): {max_diff_var_mt:.6g}")


if __name__ == "__main__":
    main()
