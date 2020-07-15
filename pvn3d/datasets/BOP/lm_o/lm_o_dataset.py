from pvn3d.common import Config
from pvn3d.lib.utils.basic_utils import Basic_Utils
import numpy as np

class Lm_o_dataset():

    def __init__(self, dataset_name, cls_type):
        self.config = Config(dataset_name='linemod', cls_type=cls_type)
        self.dataset_name = dataset_name
        self.bs_utils = Basic_Utils(self.config)


        if dataset_name == 'train':
            self.add_noise = True
            training_dataset_path = '/media/yumi/Datas/6D_Dataset/BOP_Dataste/lm-o/train_pbr/000000/depth'
            self.train_data_list = self.bs_utils.get_list(training_dataset_path)

        else:
            self.add_noise = False

        print("{}_dataset_size: ".format(dataset_name))

    def get_random_idx(self):
        n_images = len(self.train_data_list)
        idx = np.random.randint(0, n_images)
        pth = self.train_data_list[idx]
        idx = pth[0].find('.')
        pth = pth[0][0:idx]
        return pth



    def __getitem__(self, item):
        if self.dataset_name == 'train':
            item_name = self.get_random_idx()
            data =



def main():
    global DEBUG
    DEBUG = True

    cls = "duck"




if __name__ == "__main__":
    main()