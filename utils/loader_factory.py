from dataloaders.dataset import BaseDataSets, RandomGenerator, ResizeGenerator
from torchvision import transforms
from torch.utils.data import DataLoader

import random


def worker_init_fn(worker_id):
    return random.seed(worker_id)


def load(args):
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        fold=args.fold,
        sup_type=args.sup_type,
    )
    db_val = BaseDataSets(
        base_dir=args.root_path,
        split="val",
        transform=transforms.Compose([ResizeGenerator(args.patch_size)]),
        fold=args.fold,
        sup_type=args.sup_type,
    )

    train_loader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        db_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader
