# skynet-ddp-slurm-example


Two ways to run this:

```
salloc --ntasks-per-node=2 --gres=gpu:2 --nodes=1 bash -l
bash launcher.sh
```

```
sbatch launcher.sh
```


You can change the number of GPUs and number of nodes to whatever suits your fancy, but make sure that `--ntasks-per-node` is always
set to the number of GPUs you reservered.


You can run on more than 1 nodes by setting `--nodes` to some value greater than 1.
Ideally, you shouldn't use multiple nodes until you need >8 GPUs.


## Some warnings about NCCL

[NCCL](https://developer.nvidia.com/nccl) is the workhorse responsible for averaging the gradients between all the different processes
and it is really quite fast and scales incredibly well.  However, NCCL has an interesting design choice: it has no timeout functionality.
This means that once a NCCL operation begins, it either finishes, or the process running it hangs forever and because NCCL operations are CUDA
kernel, they do not respect things you'd normally expect, like the process being killed!

This has two major implications

1. When debugging, either use only 1 process or switch the backend to PyTorch's GLOO.  See [here](https://pytorch.org/docs/stable/distributed.html?highlight=init_pr#torch.distributed.init_process_group)
and [here](https://github.com/erikwijmans/skynet-ddp-slurm-example/blob/master/ddp_example/ddp_utils.py#L29).  Ideally, debug with both GLOO and 1 process!
1. Never use `scancel <job_id>` to cancel a job. Any processes not in a NCCL operation will exit immediately and the processes within a NCCL operation will
hang infinitely!  This example shows how to add signal handlers such that a job will exit cleanly when you send `SIGURS2`, which can be sent to all processes in the job via`scancel --signal USR2 <job_id>`.
