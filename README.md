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
