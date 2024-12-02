import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from src.data import PretrainDataset
from src.models import S3RecModel
from src.utils import Setting, EarlyStopping, get_item2attribute_json, get_user_seqs_long, check_path
from src.train import PretrainTrainer


# TODO: 모듈화 후 `main.py`에 편입
def run_pretrain(args):
    args.checkpoint_path = os.path.join(args.train.ckpt_path, "Pretrain.pt")

    args.data_file = args.dataset.data_path + "train_ratings.csv"
    item2attribute_file = args.dataset.data_path + args.dataset.data_name + "_item2attributes.json"
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute

    model = S3RecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, None, args)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    for epoch in range(args.pretrain.epochs):

        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pretrain.batch_size
        )

        losses = trainer.pretrain(epoch, pretrain_dataloader)

        ## comparing `sp_loss_avg``
        early_stopping(np.array([-losses["sp_loss_avg"]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
