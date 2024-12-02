import argparse
import os
from torch.utils.data import DataLoader, SequentialSampler

from src.data import SASRecDataset
from src.models import S3RecModel
from src.utils import Setting, get_item2attribute_json, get_user_seqs, generate_submission_file, check_path
from src.train import FinetuneTrainer


# TODO: 모듈화 후 `main.py`에 편입
def inference(args):
    args.data_file = args.dataset.data_path + "train_ratings.csv"
    item2attribute_file = args.dataset.data_path + args.dataset.data_name + "_item2attributes.json"

    user_seq, max_item, _, _, submission_rating_matrix = get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model}-{args.dataset.data_name}"

    print(str(args))

    args.item2attribute = item2attribute

    # args.train_matrix = submission_rating_matrix

    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
    submission_sampler = SequentialSampler(submission_dataset)
    submission_dataloader = DataLoader(
        submission_dataset, sampler=submission_sampler, batch_size=args.dataloader.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args, submission_rating_matrix)

    trainer.load(args.checkpoint_path)
    print(f"Load model from {args.checkpoint_path} for submission!")
    preds = trainer.submission(0)

    generate_submission_file(args.data_file, preds)
