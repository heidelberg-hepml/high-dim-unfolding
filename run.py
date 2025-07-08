import fastjet_contribs  # needed here to import fastjet contribs in the other files
import hydra
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from experiments.multiplicity.experiment import MultiplicityExperiment
from experiments.kinematics.experiment import KinematicsExperiment


@hydra.main(config_path="config", config_name="constituents", version_base=None)
def main(cfg):
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size > 1:
        assert torch.cuda.is_available(), "Distributed training only supported on GPU"
        # multiple GPUs -> spawn processes
        mp.spawn(ddp_worker, nprocs=world_size, args=(cfg, world_size))
    else:
        # no CPU or only one GPU -> run on main process
        ddp_worker(rank=0, cfg=cfg, world_size=world_size)


def ddp_worker(rank, cfg, world_size):
    if world_size > 1:
        # set up communication between processes
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:4242",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)

    if cfg.exp_type == "multiplicity":
        constructor = MultiplicityExperiment
    elif cfg.exp_type == "constituents" or cfg.exp_type == "jets":
        constructor = KinematicsExperiment
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp = constructor(cfg, rank, world_size)
    exp()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
