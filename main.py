import argparse
import ast
from dotenv import load_dotenv
import os
from omegaconf import OmegaConf
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module

from inference import inference
from run_pretrain import run_pretrain
from run_train import run_train
# import src.models as model_module
from src.utils import Setting, Logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type=str, help='Set configuration file path', required=True)
    parser.add_argument('--model', '-m', type=str, choices=['S3Rec'], help='Set model')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Set device')
    parser.add_argument('--seed', type=int, help='Set seed')
    parser.add_argument('--wandb', type=ast.literal_eval, help='Set wandb usage')
    parser.add_argument('--wandb-project', type=str, help='Set wandb project name')
    parser.add_argument('--wandb-entity', type=str, help='Set wandb entity name')
    parser.add_argument('--run-name', type=str, help='Set wandb run name')
    parser.add_argument('--predict', type=ast.literal_eval, help='Set predict mode')
    parser.add_argument('--checkpoint', type=str, help='Set checkpoint path')

    parser.add_argument('--model-args', type=ast.literal_eval, help='Set model arguments')
    parser.add_argument('--dataloader', type=ast.literal_eval, help='Set dataloader arguments')
    parser.add_argument('--dataset', type=ast.literal_eval, help='Set dataset arguments')
    parser.add_argument('--optimizer', type=ast.literal_eval, help='Set optimizer arguments')
    parser.add_argument('--loss', type=str, help='Set loss function')
    parser.add_argument('--lr-scheduler', type=ast.literal_eval, help='Set lr scheduler arguments')
    parser.add_argument('--metrics', type=ast.literal_eval, help='Set metrics')
    parser.add_argument('--train', type=ast.literal_eval, help='Set train arguments')

    return parser.parse_args()


# TODO: main 함수 구현
# 1. dataloader 작성
# 2. `run_train.py`, `run_pretrain.py`, `inference.py` 삭제
# 3. `utils.py/generate_submission_file` -> `setting.py/get_submit_filename` 중복 제거
# 4. `utils.py/check_path` -> `setting.py/make_dir` 중복 제거
# 5. `pretrain...` `finetune...` 쳐내기
# 6. `RecBole` 적용
def main(args):
    Setting.set_seed(args.seed)
    setting = Setting()

    run_train(args)
    run_pretrain(args)
    inference(args)


if __name__ == '__main__':
    load_dotenv()
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY')

    args = parse_args()

    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    if not config_yaml.predict:
        del config_yaml.checkpoint

        if not config_yaml.wandb:
            del config_yaml.wandb_project, config_yaml.run_name

        # config_yaml.model_args = OmegaConf.create({config_yaml.model: config_yaml.model_args[config_yaml.model]})

        # config_yaml.optimizer.args = {k: v for k, v in config_yaml.optimizer.args.items()
        #                               if k in getattr(optimizer_module,
        #                                               config_yaml.optimizer.type).__init__.__code__.co_varnames}

        if not config_yaml.lr_scheduler.use:
            del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        else:
            config_yaml.lr_scheduler.args = {k: v for k, v in config_yaml.lr_scheduler.args.items()
                                             if k in getattr(scheduler_module,
                                                             config_yaml.lr_scheduler.type).__init__.__code__.co_varnames}

        if not config_yaml.train.resume:
            del config_yaml.train.resume_path

    print(OmegaConf.to_yaml(config_yaml))

    if args.wandb or config_yaml.wandb:
        import wandb

        wandb.init(
            entity=config_yaml.wandb_entity,
            project=config_yaml.wandb_project,
            config=OmegaConf.to_container(config_yaml, resolve=True),
            name=config_yaml.run_name if config_yaml.run_name else None,
            notes=config_yaml.memo if hasattr(config_yaml, 'memo') else None,
            tags=[config_yaml.model],
            resume="allow"
        )

        config_yaml.run_href = wandb.run.get_url()

        wandb.run.log_code("./src")

    main(config_yaml)

    if args.wandb:
        wandb.finish()
