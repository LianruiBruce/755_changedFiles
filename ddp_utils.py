import os
import shutil

import torch
import torch.distributed as dist


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    # 没初始化分布式就直接返回
    if not dist.is_available() or not dist.is_initialized():
        return
    for param in params:
        dist.broadcast(param.data, src=0)


def init_processes(rank, size, fn, args):
    """
    Initialize (if needed) the distributed environment and run fn.
    - 单卡/单进程: 直接调用 fn，不走分布式
    - Windows 或 NCCL 不可用: 自动使用 gloo
    """
    # 兜底 master 地址/端口
    os.environ.setdefault("MASTER_ADDR", getattr(args, "master_address", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", str(getattr(args, "master_port", "29500")))

    # 单卡/单进程：跳过 DDP
    if size is None or size <= 1 or getattr(args, "num_process_per_node", 1) <= 1:
        gpu = getattr(args, "local_rank", 0)
        if torch.cuda.is_available() and 0 <= gpu < torch.cuda.device_count():
            torch.cuda.set_device(gpu)
        return fn(rank, gpu, args)

    # 多进程：选择后端
    backend = "nccl"
    use_nccl = False
    if os.name != "nt":  # 非 Windows 才考虑 nccl
        try:
            use_nccl = hasattr(dist, "is_nccl_available") and dist.is_nccl_available()
        except Exception:
            use_nccl = False
    if not use_nccl:
        backend = "gloo"
        # 避免 "use_libuv was requested but PyTorch was built without libuv support"
        os.environ.setdefault("TORCH_USE_LIBUV", "0")

    gpu = getattr(args, "local_rank", rank)
    if torch.cuda.is_available() and 0 <= gpu < torch.cuda.device_count():
        torch.cuda.set_device(gpu)

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=size)
    try:
        fn(rank, gpu, args)
        dist.barrier()
    finally:
        cleanup()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
