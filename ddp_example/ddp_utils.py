import os

import ifcfg
import torch
import torch.distributed as distrib
import torch.nn as nn


def get_ifname():
    return ifcfg.default_interface()["device"]


def init_distrib_slurm(backend="nccl"):
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    tcp_store = distrib.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    distrib.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store


def convert_groupnorm_model(module, ngroups=32):
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(ngroups, module.num_features, affine=module.affine)
    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm_model(child, ngroups))

    return mod
