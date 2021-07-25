# skynet-ddp-slurm-example


Two ways to run this:

```
salloc --ntasks-per-node=2 --gpus-per-task=1 --nodes=1 bash -l
bash launcher.sh
```

```
sbatch launcher.sh
```


You can change the number of processes (tasks) and number of nodes to whatever suits your fancy.  Slurm will always give you 1 GPU per process thanks to `--gpus-per-task=1`


You can run on more than 1 nodes by setting `--nodes` to some value greater than 1.
Ideally, you shouldn't use multiple nodes until you need >8 GPUs.  Note that while multi-node does work on skynet, the interconnect between nodes is quite slow (10 gig ethernet -- while that may sound fast, "fast" interconnects by HPC standards are 400+ gig infiniband), so 2 nodes may be slower than 1. 


## Some warnings about NCCL

**Update**: More recent versions of NCCL have the ability to monitor themselves and not spin forever.  When debugging, do `export NCCL_BLOCKING_WAIT=1` or `export NCCL_ASYNC_ERROR_HANDLING=1`.  The latter has a much lower performance overhead but makes crashes more opaque.

[NCCL](https://developer.nvidia.com/nccl) is the workhorse responsible for averaging the gradients between all the different processes
and it is really quite fast and scales incredibly well.  However, NCCL has an interesting design choice: it has no timeout functionality.
This means that once a NCCL operation begins, it either finishes, or the process running it hangs forever and because NCCL operations are CUDA
kernel, they do not respect things you'd normally expect, like the process being killed!

This has two major implications

1. When debugging, either use only 1 process or switch the backend to PyTorch's GLOO.  See [here](https://pytorch.org/docs/stable/distributed.html?highlight=init_pr#torch.distributed.init_process_group)
and [here](https://github.com/erikwijmans/skynet-ddp-slurm-example/blob/master/ddp_example/ddp_utils.py#L29).  Ideally, debug with both GLOO and 1 process!
1. Use `scancel <job_id>` to cancel a job sparringly. Any processes not in a NCCL operation will exit immediately and the processes within a NCCL operation will
hang infinitely!  This example shows how to add signal handlers such that a job will exit cleanly when you send `SIGURS2`, which can be sent to all processes in the job via`scancel --signal USR2 <job_id>`.
