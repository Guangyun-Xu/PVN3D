import os
import yaml
import numpy as np
from pvn3d.lib.utils.basic_utils import Basic_Utils

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

class Config(object):
    def __init__(self, train_list_dir, test_list_dir):
        self.bs_utils = Basic_Utils()

        self.dataset_name = 'bop'
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)
        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(
                self.exp_dir,
                'lib/ResNet_pretrained_mdl'
            )
        )
        ensure_fd(self.resnet_ptr_mdl_p)

        # log folder
        self.log_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', self.dataset_name)
        )
        ensure_fd(self.log_dir)
        self.log_model_dir = os.path.join(self.log_dir, 'checkpoints')
        ensure_fd(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.log_dir, 'eval_results')
        ensure_fd(self.log_eval_dir)

        self.train_list = self.bs_utils.read_lines(train_list_dir)
        self.test_list = self.bs_utils.read_lines(test_list_dir)
        self.n_train_frame = len(self.train_list)
        self.n_test_frame = len(self.test_list)

        self.n_total_epoch =10
        self.mini_batch_size = 2
        self.num_mini_batch_per_epoch = self.n_train_frame
        self.val_mini_batch_size = 2
        self.val_num_mini_batch_per_epoch = self.n_test_frame
        self.test_mini_batch_size = 4

        self.voxel_size = 0.01  # 以米为单位
        self.n_sample_points = 20000