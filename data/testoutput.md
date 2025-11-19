目录说明:
╔═════════════════╦════════╦════╦═════════════════════════════════════════════════════════════════════════╗
║目录             ║名称    ║速度║说明                                                                     ║
╠═════════════════╬════════╬════╬═════════════════════════════════════════════════════════════════════════╣
║/                ║系 统 盘║一般║实例关机数据不会丢失，可存放代码等。会随保存镜像一起保存。               ║
║/root/tmp        ║数 据 盘║ 快 ║实例关机数据不会丢失，可存放读写IO要求高的数据。但不会随保存镜像一起保存 ║
╚═════════════════╩════════╩════╩═════════════════════════════════════════════════════════════════════════╝
CPU ：10 核心
内存：120 GB
GPU ：NVIDIA GeForce RTX 4090, 1
存储：
  系 统 盘/               ：62% 19G/30G
  数 据 盘/root/lanyun-tmp：1% 584K/150G
+----------------------------------------------------------------------------------------------------------------+
*注意: 系统盘较小请将大的数据存放于数据盘或网盘中，重置系统时数据盘和网盘中的数据不受影响
root@76fafe984844:~# conda env list
# conda environments:
#
base                     /root/miniconda
cudf_test                /root/miniconda/envs/cudf_test
rapids-25.10             /root/miniconda/envs/rapids-25.10

root@76fafe984844:~# conda create -n  vec_test
Retrieving notices: ...working... done
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /root/miniconda/envs/vec_test



Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate vec_test
#
# To deactivate an active environment, use
#
#     $ conda deactivate

root@76fafe984844:~# conda activate vec_test
(vec_test) root@76fafe984844:~# cd ~/vec_test
bash: cd: /root/vec_test: No such file or directory
(vec_test) root@76fafe984844:~# cd lanyun-tmp/vec_test/
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#  python benchmark/generate_csv.py
Generated CSV: data/prices_3000x5000.csv  shape=(5000, 3000)
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 100.5052 s
[GPU] MA window=20 on cuda: 0.2642 s
Speedup (CPU / GPU) = 380.43x
Max abs diff between CPU and GPU MA (ignoring NaN): 9.15527e-05
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#   python benchmark/benchmark_ma.py \
    --csv data/prices_3000x5000.csv \
    --window 20 \
    --device cuda \
    --threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 83.9040 s
[CPU-mt] MA window=20, threads=16: 14.0700 s
Traceback (most recent call last):
  File "/root/lanyun-tmp/vec_test/benchmark/benchmark_ma.py", line 165, in <module>
    main()
  File "/root/lanyun-tmp/vec_test/benchmark/benchmark_ma.py", line 143, in main
    ma_gpu, gpu_time, gpu_mem = ma_gpu_conv_all(prices, args.window, device=args.device)
  File "/root/lanyun-tmp/vec_test/benchmark/benchmark_ma.py", line 87, in ma_gpu_conv_all
    torch.cuda.set_device(dev)
  File "/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 397, in set_device
    device = _get_device_index(device)
  File "/root/miniconda/lib/python3.10/site-packages/torch/cuda/_utils.py", line 38, in _get_device_index
    return _torch_get_device_index(device, optional, allow_cpu)
  File "/root/miniconda/lib/python3.10/site-packages/torch/_utils.py", line 798, in _get_device_index
    raise ValueError(
ValueError: Expected a torch.device with a specified index or an integer, but got:cuda
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py \                                                        
    --csv data/prices_3000x5000.csv \
    --window 20 \
    --device cuda:0 \
    --threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 86.6148 s
[CPU-mt] MA window=20, threads=16: 12.8091 s
[GPU] MA window=20 on cuda:0: 0.1601 s
[GPU] Peak memory allocated: 174.22 MB
Speedup (CPU-serial / GPU) = 541.02x
Speedup (CPU-mt / GPU)      = 80.01x
Speedup (CPU-serial / CPU-mt) = 6.76x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#  python benchmark/benchmark_ma.py  --csv data/prices_3000x5000.csv --window 20 --device cuda --threads 16
^Z
[1]+  Stopped                 python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda --threads 16
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py  --csv data/prices_3000x5000.csv --window 20 --device cuda:0 --threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4149 s
[CPU-mt] MA window=20, threads=16: 0.2711 s
[GPU] MA window=20 on cuda:0: 0.1330 s
[GPU] Peak memory allocated: 174.22 MB
Speedup (CPU-serial / GPU) = 3.12x
Speedup (CPU-mt / GPU)      = 2.04x
Speedup (CPU-serial / CPU-mt) = 1.53x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ema.py --csv data/prices_3000x5000.csv --span   20 --device cuda:0 --threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] EMA span=20: 0.4750 s
[CPU-mt] EMA span=20, threads=16: 0.7275 s
[GPU] EMA span=20 on cuda:0: 0.1921 s
[GPU] Peak memory allocated: 174.22 MB
Speedup (CPU-serial / GPU)   = 2.47x
Speedup (CPU-mt / GPU)       = 3.79x
Speedup (CPU-serial / CPU-mt) = 0.65x
Max abs diff between CPU-serial and GPU EMA (ignoring NaN): 6.10352e-05
Max abs diff between CPU-serial and CPU-mt EMA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ema.py --csv data/prices_3000x5000.csv --span   20 --device cpu -
-threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] EMA span=20: 0.5635 s
[CPU-mt] EMA span=20, threads=16: 0.9980 s
[GPU] EMA span=20 on cpu: 0.5070 s
Speedup (CPU-serial / GPU)   = 1.11x
Speedup (CPU-mt / GPU)       = 1.97x
Speedup (CPU-serial / CPU-mt) = 0.56x
Max abs diff between CPU-serial and GPU EMA (ignoring NaN): 0
Max abs diff between CPU-serial and CPU-mt EMA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py  --csv data/prices_3000x5000.csv --window 20 --device cpu -
-threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4223 s
[CPU-mt] MA window=20, threads=16: 0.2924 s
[GPU] MA window=20 on cpu: 0.4715 s
Speedup (CPU-serial / GPU) = 0.90x
Speedup (CPU-mt / GPU)      = 0.62x
Speedup (CPU-serial / CPU-mt) = 1.44x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#  python benchmark/benchmark_rolling_std_var.py \
    --csv data/prices_3000x5000.csv \
    --window 20 \
    --device cuda:0 \
    --threads 16
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] rolling window=20: 1.1430 s
[CPU-mt] rolling window=20, threads=16: 0.6546 s
[GPU] rolling window=20 on cuda:0: 0.1987 s
[GPU] Peak memory allocated: 290.22 MB
Speedup (CPU-serial / GPU)    = 5.75x
Speedup (CPU-mt / GPU)        = 3.29x
Speedup (CPU-serial / CPU-mt) = 1.75x
Max abs diff Std (CPU-serial vs GPU, ignoring NaN): 0.0412075
Max abs diff Std (CPU-serial vs CPU-mt, ignoring NaN): 0
Max abs diff Var (CPU-serial vs GPU, ignoring NaN): 0.0794754
Max abs diff Var (CPU-serial vs CPU-mt, ignoring NaN): 0



11/19
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4048 s
[CPU-mt] MA window=20, threads=80: 0.4752 s
[GPU] MA window=20 on cuda:0: 0.0177 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 22.86x
Speedup (CPU-mt / GPU)      = 26.84x
Speedup (CPU-serial / CPU-mt) = 0.85x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4034 s
[CPU-mt] MA window=20, threads=80: 0.4499 s
<module 'pynvml' from '/root/miniconda/lib/python3.10/site-packages/pynvml.py'>
[GPU] MA window=20 on cuda:0: 0.0162 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 24.90x
Speedup (CPU-mt / GPU)      = 27.77x
Speedup (CPU-serial / CPU-mt) = 0.90x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.3878 s
[CPU-mt] MA window=20, threads=80: 0.4863 s
False
[GPU] MA window=20 on cuda:0: 0.0157 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 24.63x
Speedup (CPU-mt / GPU)      = 30.89x
Speedup (CPU-serial / CPU-mt) = 0.80x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4041 s
[CPU-mt] MA window=20, threads=80: 0.4583 s
[GPU] MA window=20 on cuda:0: 0.0158 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 25.57x
Speedup (CPU-mt / GPU)      = 29.00x
Speedup (CPU-serial / CPU-mt) = 0.88x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# nvidia-smi
Wed Nov 19 11:12:30 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:12:00.0 Off |                  Off |
|  0%   30C    P8             12W /  450W |       0MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4004 s
[CPU-mt] MA window=20, threads=80: 0.4648 s
[GPU] MA window=20 on cuda:0: 0.0191 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 20.98x
Speedup (CPU-mt / GPU)      = 24.35x
Speedup (CPU-serial / CPU-mt) = 0.86x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4003 s
[CPU-mt] MA window=20, threads=80: 0.5054 s
[GPU] MA window=20 on cuda:0: 0.0021 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 192.33x
Speedup (CPU-mt / GPU)      = 242.82x
Speedup (CPU-serial / CPU-mt) = 0.79x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4075 s
[CPU-mt] MA window=20, threads=80: 0.4462 s
[GPU] MA window=20 on cuda:0: 0.0020 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 201.52x
Speedup (CPU-mt / GPU)      = 220.67x
Speedup (CPU-serial / CPU-mt) = 0.91x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4005 s
[CPU-mt] MA window=20, threads=80: 0.4503 s
[GPU] MA window=20 on cuda:0: 0.0021 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 192.48x
Speedup (CPU-mt / GPU)      = 216.39x
Speedup (CPU-serial / CPU-mt) = 0.89x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4105 s
[CPU-mt] MA window=20, threads=80: 0.4182 s
[GPU] MA window=20 on cuda:0: 0.0025 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 163.01x
Speedup (CPU-mt / GPU)      = 166.05x
Speedup (CPU-serial / CPU-mt) = 0.98x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.3917 s
[CPU-mt] MA window=20, threads=80: 0.4313 s
[GPU] MA window=20 on cuda:0: 0.0014 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 285.26x
Speedup (CPU-mt / GPU)      = 314.10x
Speedup (CPU-serial / CPU-mt) = 0.91x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4050 s
[CPU-mt] MA window=20, threads=80: 0.4373 s
[GPU] MA window=20 on cuda:0: 0.0014 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 299.20x
Speedup (CPU-mt / GPU)      = 323.05x
Speedup (CPU-serial / CPU-mt) = 0.93x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4132 s
[CPU-mt] MA window=20, threads=80: 0.3961 s
[GPU] MA window=20 on cuda:0: 0.0018 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 227.15x
Speedup (CPU-mt / GPU)      = 217.76x
Speedup (CPU-serial / CPU-mt) = 1.04x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4288 s
[CPU-mt] MA window=20, threads=80: 0.4703 s
[GPU] MA window=20 on cuda:0: 0.0016 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 266.83x
Speedup (CPU-mt / GPU)      = 292.69x
Speedup (CPU-serial / CPU-mt) = 0.91x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ema.py --csv data/prices_3000x5000.csv --window 20 --device cuda:
0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
usage: benchmark_ema.py [-h] [--csv CSV] [--span SPAN] [--threads THREADS] [--device DEVICE]
benchmark_ema.py: error: unrecognized arguments: --window 20
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ema.py --csv data/prices_3000x5000.csv --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] EMA span=20: 0.6484 s
[CPU-mt] EMA span=20, threads=80: 0.8051 s
[GPU] EMA span=20 on cuda:0: 0.0031 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU)   = 209.78x
Speedup (CPU-mt / GPU)       = 260.49x
Speedup (CPU-serial / CPU-mt) = 0.81x
Max abs diff between CPU-serial and GPU EMA (ignoring NaN): 6.10352e-05
Max abs diff between CPU-serial and CPU-mt EMA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4069 s
[CPU-mt] MA window=20, threads=80: 0.4312 s
[GPU] MA window=20 on cuda:0: 0.0008 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 526.68x
Speedup (CPU-mt / GPU)      = 558.17x
Speedup (CPU-serial / CPU-mt) = 0.94x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4056 s
[CPU-mt] MA window=20, threads=80: 0.4811 s
[GPU] MA window=20 on cuda:0: 0.0008 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 497.38x
Speedup (CPU-mt / GPU)      = 589.88x
Speedup (CPU-serial / CPU-mt) = 0.84x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.3981 s
[CPU-mt] MA window=20, threads=80: 0.4116 s
[GPU] MA window=20 on cuda:0: 0.0008 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 33.33%
Speedup (CPU-serial / GPU) = 525.53x
Speedup (CPU-mt / GPU)      = 543.39x
Speedup (CPU-serial / CPU-mt) = 0.97x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4223 s
[CPU-mt] MA window=20, threads=80: 0.4393 s
0.7335989261046052
[GPU] MA window=20 on cuda:0: 0.0007 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 5.26%
Speedup (CPU-serial / GPU) = 575.62x
Speedup (CPU-mt / GPU)      = 598.88x
Speedup (CPU-serial / CPU-mt) = 0.96x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4049 s
[CPU-mt] MA window=20, threads=80: 0.4851 s
0.22639438370242715
[GPU] MA window=20 on cuda:0: 0.0023 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 0.00%
Speedup (CPU-serial / GPU) = 178.83x
Speedup (CPU-mt / GPU)      = 214.26x
Speedup (CPU-serial / CPU-mt) = 0.83x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# sudo apt install nvtop
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following packages were automatically installed and are no longer required:
  javascript-common libc-ares2 libjs-highlight.js libnode72
Use 'apt autoremove' to remove them.
The following NEW packages will be installed:
  nvtop
0 upgraded, 1 newly installed, 0 to remove and 105 not upgraded.
Need to get 43.9 kB of archives.
After this operation, 106 kB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 nvtop amd64 1.2.2-1 [43.9 kB]
Fetched 43.9 kB in 1s (36.5 kB/s)
debconf: delaying package configuration, since apt-utils is not installed
Selecting previously unselected package nvtop.
(Reading database ... 41834 files and directories currently installed.)
Preparing to unpack .../nvtop_1.2.2-1_amd64.deb ...
Unpacking nvtop (1.2.2-1) ...
Setting up nvtop (1.2.2-1) ...
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4039 s
[CPU-mt] MA window=20, threads=80: 0.4800 s
0.2559018451720476
[GPU] MA window=20 on cuda:0: 0.0026 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 1.67%
Speedup (CPU-serial / GPU) = 157.85x
Speedup (CPU-mt / GPU)      = 187.56x
Speedup (CPU-serial / CPU-mt) = 0.84x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.4190 s
[CPU-mt] MA window=20, threads=80: 0.4368 s
1.3171815038658679
[GPU] MA window=20 on cuda:0: 0.0007 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 27.59%
Speedup (CPU-serial / GPU) = 636.27x
Speedup (CPU-mt / GPU)      = 663.24x
Speedup (CPU-serial / CPU-mt) = 0.96x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# python benchmark/benchmark_ma.py --csv data/prices_3000x5000.csv --window 20 --device cuda:0
/root/miniconda/lib/python3.10/site-packages/torch/cuda/__init__.py:56: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loaded prices from data/prices_3000x5000.csv, shape = (T=5000, S=3000)
[CPU] MA window=20: 0.3975 s
[CPU-mt] MA window=20, threads=80: 0.4587 s
0.7463628887198865
[GPU] MA window=20 on cuda:0: 0.0007 s
[GPU] Peak memory allocated: 232.22 MB
[GPU] Average utilization: 27.78%
Speedup (CPU-serial / GPU) = 532.54x
Speedup (CPU-mt / GPU)      = 614.55x
Speedup (CPU-serial / CPU-mt) = 0.87x
Max abs diff between CPU-serial and GPU MA (ignoring NaN): 9.15527e-05
Max abs diff between CPU-serial and CPU-mt MA (ignoring NaN): 0
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git init
hint: Using 'master' as the name for the initial branch. This default branch name
hint: is subject to change. To configure the initial branch name to use in all
hint: of your new repositories, which will suppress this warning, call:
hint: 
hint:   git config --global init.defaultBranch <name>
hint: 
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint: 
hint:   git branch -m <name>
Initialized empty Git repository in /root/lanyun-tmp/torch_test/.git/
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git add .
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#  git commit -m "gpu 0.008s"
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'root@76fafe984844.(none)')
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git config --global user.email "qaqxhy@qq.com"
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git config --global user.name "qwqxhy"
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git commit -m "gpu 0.008s"
[master (root-commit) 341fe01] gpu 0.008s
 13 files changed, 6057 insertions(+)
 create mode 100644 __pycache__/ema.cpython-310.pyc
 create mode 100644 __pycache__/gpu_utilization.cpython-310.pyc
 create mode 100644 __pycache__/ma.cpython-310.pyc
 create mode 100644 benchmark/benchmark_ema.py
 create mode 100644 benchmark/benchmark_ma.py
 create mode 100644 benchmark/benchmark_rolling_std_var.py
 create mode 100644 benchmark/generate_csv.py
 create mode 100644 data/prices_3000x5000.csv
 create mode 100644 data/test_prices.csv
 create mode 100644 data/testoutput.md
 create mode 100644 ema.py
 create mode 100644 gpu_utilization.py
 create mode 100644 ma.py
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#   git branch -M main
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git remote add origin https://github.com/qwqxhy/torch_test.git
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#  git push -u origin main
Enumerating objects: 18, done.
Counting objects: 100% (18/18), done.
Delta compression using up to 80 threads
Compressing objects: 100% (18/18), done.
Writing objects: 100% (18/18), 152.75 MiB | 3.97 MiB/s, done.
Total 18 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), done.
remote: error: Trace: f8b64f1d0aee1143863f2566f123b0adc642b37f3815771fe3db5c9db0c47ad7
remote: error: See https://gh.io/lfs for more information.
remote: error: File data/prices_3000x5000.csv is 358.05 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
To https://github.com/qwqxhy/torch_test.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/qwqxhy/torch_test.git'
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git rm data/prices_3000x5000.csv
rm 'data/prices_3000x5000.csv'
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# echo "data/prices_3000x5000.csv" >> .gitignore
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git add .gitignore
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# git commit --amend --no-edit
[main 4b3dafa] gpu 0.008s
 Date: Wed Nov 19 15:59:22 2025 +0800
 13 files changed, 1057 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 __pycache__/ema.cpython-310.pyc
 create mode 100644 __pycache__/gpu_utilization.cpython-310.pyc
 create mode 100644 __pycache__/ma.cpython-310.pyc
 create mode 100644 benchmark/benchmark_ema.py
 create mode 100644 benchmark/benchmark_ma.py
 create mode 100644 benchmark/benchmark_rolling_std_var.py
 create mode 100644 benchmark/generate_csv.py
 create mode 100644 data/test_prices.csv
 create mode 100644 data/testoutput.md
 create mode 100644 ema.py
 create mode 100644 gpu_utilization.py
 create mode 100644 ma.py
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test#  git push -u origin main
Enumerating objects: 18, done.
Counting objects: 100% (18/18), done.
Delta compression using up to 80 threads
Compressing objects: 100% (17/17), done.
Writing objects: 100% (18/18), 15.78 KiB | 2.25 MiB/s, done.
Total 18 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), done.
To https://github.com/qwqxhy/torch_test.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
(vec_test) root@76fafe984844:~/lanyun-tmp/vec_test# 