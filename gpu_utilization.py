import threading
import time
from typing import Optional

# Try to import the newer nvidia-ml-py bindings first, then fall back to pynvml.
_nvml = None
try:  # pragma: no cover - import resolution logic
    import nvidia_ml_py as _nvml  # type: ignore
except Exception:
    try:
        import pynvml as _nvml  # type: ignore
    except Exception:
        _nvml = None

try:  # Optional: align NVML index with PyTorch CUDA device index via PCI bus id
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional for this helper
    torch = None  # type: ignore


class GpuUtilizationMonitor:
    """
    Lightweight helper that samples GPU utilization (%) periodically
    using NVML (via nvidia-ml-py / pynvml) in a background thread.

    If NVML is not available or NVML init fails, the monitor becomes
    a no-op and stop() will return None.
    """

    def __init__(self, device_index: int, interval_sec: float = 0.05) -> None:
        self._device_index = device_index
        self._interval_sec = interval_sec
        self._samples: list[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._enabled = _nvml is not None

    def start(self) -> None:
        if not self._enabled:
            return

        try:
            _nvml.nvmlInit()  # type: ignore[attr-defined]

            # Map PyTorch CUDA device index -> NVML device index using PCI bus id,
            # to correctly handle CUDA_VISIBLE_DEVICES / container remapping.
            nvml_index = self._device_index
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(self._device_index)  # type: ignore[attr-defined]
                        target_bus_id = getattr(props, "pci_bus_id", None)
                        if target_bus_id:
                            count = _nvml.nvmlDeviceGetCount()  # type: ignore[attr-defined]
                            for i in range(count):
                                handle = _nvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[attr-defined]
                                pci_info = _nvml.nvmlDeviceGetPciInfo(handle)  # type: ignore[attr-defined]
                                bus_id = getattr(pci_info, "busId", None)
                                if isinstance(bus_id, bytes):
                                    bus_id = bus_id.decode()
                                if bus_id == target_bus_id:
                                    nvml_index = i
                                    break
                except Exception:
                    # If anything goes wrong, fall back to original index.
                    pass

            self._handle = _nvml.nvmlDeviceGetHandleByIndex(nvml_index)  # type: ignore[attr-defined]
        except Exception:
            # If NVML is not usable, disable monitoring silently.
            self._enabled = False
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                util = _nvml.nvmlDeviceGetUtilizationRates(self._handle)  # type: ignore[attr-defined]
                self._samples.append(float(util.gpu))
            except Exception:
                # Ignore transient NVML errors; keep best-effort sampling.
                pass
            time.sleep(self._interval_sec)

    def stop(self) -> Optional[float]:
        """
        Stop sampling and return the average GPU utilization in percent.
        Returns None if monitoring was not enabled or produced no samples.
        """
        if not self._enabled:
            return None
        if not self._running:
            return None

        self._running = False
        if self._thread is not None:
            self._thread.join()

        try:
            _nvml.nvmlShutdown()  # type: ignore[attr-defined]
        except Exception:
            # Shutdown errors are not critical for our benchmark.
            pass

        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)
