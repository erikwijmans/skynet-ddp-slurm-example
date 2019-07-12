# skynet-ddp-slurm-example


Two ways to run this:

```
salloc --ntasks-per-node=2 --gres=gpu:2 --nodes=2 bash -l
bash launcher.sh
```

```
sbatch launcher.sh
```
