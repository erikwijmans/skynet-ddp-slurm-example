import torch
import torch.distributed as distrib
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import tqdm

from ddp_example.ddp_utils import convert_groupnorm_model, init_distrib_slurm

BATCH_SIZE = 128
LR = 1e-2
WD = 5e-3


def train_epoch(model, optimizer, dloader, epoch):
    device = next(model.parameters()).device

    train_stats = torch.zeros((3,), device=device)
    for batch in dloader:
        batch = tuple(v.to(device) for v in batch)
        x, y = batch

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        optimizer.zero_grad()
        (loss / x.size(0)).backward()
        optimizer.step()

        train_stats[0] += loss
        train_stats[1] += (torch.argmax(logits, -1) == y).float().sum()
        train_stats[2] += x.size(0)

    distrib.all_reduce(train_stats)

    if distrib.get_rank() == 0:
        train_stats /= train_stats[2]
        tqdm.tqdm.write("\n===== Epoch {:3d} =====\n".format(epoch))
        tqdm.tqdm.write(
            "Train: Loss={:.3f}    Acc={:.3f}".format(
                train_stats[0].item(), train_stats[1].item()
            )
        )


def eval_epoch(model, dloader):
    device = next(model.parameters()).device

    eval_stats = torch.zeros((3,), device=device)
    for batch in dloader:
        batch = tuple(v.to(device) for v in batch)
        x, y = batch

        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")

        eval_stats[0] += loss
        eval_stats[1] += (torch.argmax(logits, -1) == y).float().sum()
        eval_stats[2] += x.size(0)

    distrib.all_reduce(eval_stats)

    if distrib.get_rank() == 0:
        eval_stats /= eval_stats[2]
        tqdm.tqdm.write(
            "Val:   Loss={:.3f}    Acc={:.3f}".format(
                eval_stats[0].item(), eval_stats[1].item()
            )
        )


def main():
    local_rank, _ = init_distrib_slurm(backend="nccl")
    world_rank = distrib.get_rank()
    world_size = distrib.get_world_size()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Download CIFAR10 just on rank 0
    if world_rank == 0:
        torchvision.datasets.CIFAR10("data", download=True)

    # Have all workers wait
    distrib.barrier()

    train_dset = torchvision.datasets.CIFAR10(
        "data", train=True, transform=T.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=train_dset, num_replicas=world_size, rank=world_rank
        ),
        drop_last=True,
        num_workers=4,
    )

    val_dset = torchvision.datasets.CIFAR10("data", train=False, transform=T.ToTensor())
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=val_dset, num_replicas=world_size, rank=world_rank
        ),
        num_workers=4,
    )

    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    # Let's use group norm instead of batch norm because batch norm can be problematic if the batch size per GPU gets really small
    model = convert_groupnorm_model(model, ngroups=32)
    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=WD
    )

    for epoch in range(50):
        train_epoch(model, optimizer, train_loader, epoch)
        eval_epoch(model, val_loader)

        train_loader.sampler.set_epoch(epoch)


if __name__ == "__main__":
    main()
