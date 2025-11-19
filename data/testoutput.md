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