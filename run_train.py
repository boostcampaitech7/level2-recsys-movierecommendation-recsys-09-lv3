import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.data import SASRecDataset
from src.models import S3RecModel
from src.utils import Setting, EarlyStopping, get_item2attribute_json, get_user_seqs, check_path
from src.train import FinetuneTrainer


# TODO: 모듈화 후 `main.py`에 편입
def run_train(args):
    args.data_file = args.dataset.data_path + "train_ratings.csv"
    item2attribute_file = args.dataset.data_path + args.dataset.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model}-{args.dataset.data_name}"
    args.log_file = os.path.join(args.train.log_path, args_str + ".log")
    print(str(args))

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    # args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.train.ckpt_path, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.dataloader.batch_size
    )

    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.dataloader.batch_size
    )

    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.dataloader.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args, valid_rating_matrix
    )

    print(args.predict)
    if args.predict:
        pretrained_path = os.path.join(args.train.ckpt_path, "Pretrain.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.train.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.set_train_matrix(test_rating_matrix)
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)
