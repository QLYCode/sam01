import logging

import sys

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss


from tqdm import tqdm
import math

from networks.net_factory import net_factory

import utils.parse_args as parse_args
import utils.loader_factory as loader_factory
import utils.trainer as trainer
import utils.validator as validator
import utils.helpers as helpers


def init():
    args = parse_args.parse_args()
    helpers.set_seed(args.seed, args.deterministic)
    snapshot_path = helpers.setup_snapshot(args)
    helpers.setup_logging(args)
    return args, snapshot_path


def train():
    device = "cuda:0"
    args, snapshot_path = init()
    model = net_factory(
        net_type=args.model, in_chns=1, class_num=args.num_classes, device=device
    )
    train_loader, val_loader = loader_factory.load(args)
    # import torch
    # state_dict = torch.load(snapshot_path + "/best.pth", map_location=device, weights_only=False)
    # model.load_state_dict(state_dict)
    model.train()

    optimizer = optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss(ignore_index=4)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(train_loader)))
    train_iter_num = 0
    val_iter_num = 0
    max_epoch = args.max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = math.inf
    for _ in iterator:
        new_iter_num = trainer.train_epoch(
            train_loader, model, args, writer, ce_loss, optimizer, train_iter_num, device
        )
        train_iter_num = new_iter_num
        new_iter_num, new_loss = validator.run_epoch(
            val_loader, model, args, writer, ce_loss, best_loss, val_iter_num, device
        )
        val_iter_num = new_iter_num
        best_loss = new_loss
        if train_iter_num >= args.max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    train()
