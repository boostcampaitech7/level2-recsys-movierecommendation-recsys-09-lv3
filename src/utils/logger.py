import logging
import os
from omegaconf import OmegaConf


class Logger:
    def __init__(self, args, path):
        self.args = args
        self.path = path

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('[%(asctime)s] - %(message)s')

        self.file_handler = logging.FileHandler(os.path.join(self.path, 'train.log'))
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, epoch, train_loss, valid_loss=None, valid_metrics=None):
        message = f'epoch : {epoch}/{self.args.train.epochs} | train loss : {train_loss:.3f}'
        if valid_loss:
            message += f' | valid loss : {valid_loss:.3f}'
        if valid_metrics:
            for metric, value in valid_metrics.items():
                message += f' | valid {metric.lower()} : {value:.3f}'
        self.logger.info(message)

    def close(self):
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def save_args(self):
        with open(os.path.join(self.path, 'config.yaml'), 'w') as f:
            OmegaConf.save(self.args, f)

    def __del__(self):
        self.close()
