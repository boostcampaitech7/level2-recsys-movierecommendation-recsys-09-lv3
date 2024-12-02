import numpy as np
import os
import random
import time
import torch


class Setting:
    @staticmethod
    def set_seed(seed: int):
        """
        set seed for reproducibility
        :param seed: (int) seed
        :return: (None)
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self):
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        self.save_time = save_time

    def get_log_path(self, args):
        path = os.path.join(args.train.log_dir, f'{self.save_time}_{args.model}/')
        self.make_dir(path)

        return path

    def get_submit_filename(self, args):
        if not args.predict:
            self.make_dir(args.train.submit_dir)
            filename = os.path.join(args.train.submit_dir, f'{self.save_time}_{args.model}.csv')
        else:
            filename = os.path.basename(args.checkpoint)
            filename = os.path.join(args.train.submit_dir, f'{filename}.csv')

        return filename

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path
