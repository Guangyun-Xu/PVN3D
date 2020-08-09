import torch
from torch import nn
import torchvision.transforms as transforms
import open3d as o3d
__all__ = [o3d]
import os
import os.path
import numpy as np
import json
from pvn3d.common_BOP import Config
from pvn3d.lib.utils.basic_utils import Basic_Utils
from PIL import Image
import matplotlib.pyplot as plt



import time



class BOPDataset():
    def __init__(self, data_list_path, cls_id):
        # 方法调用
        self.config = Config(data_list_path, data_list_path)
        self.bs_utils = Basic_Utils()
        self.cls_id = cls_id
        # 参数设置
        self.voxel_size = self.config.voxel_size
        self.n_sample_points = self.config.n_sample_points
        self.add_noise = True

        # 数据获取
        self.data_list = self.bs_utils.read_lines(data_list_path)

        # 功能定义
        self.trans_color = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)



    def get_cam_parameter(self, folderPath, sceneId):
        sceneInfoPath = os.path.join(folderPath, "scene_camera.json")
        with open(sceneInfoPath, 'r') as f2:
            sceneInfo = json.load(f2)
            sceneId = sceneId.lstrip('0')
            if sceneId == '':
                sceneId = '0'
            sceneDate = sceneInfo[sceneId]
            return sceneDate


    def get_normal(self, cld):
        cloud = o3d.geometry.PointCloud()
        cld = cld.astype(np.float32)
        cloud.points = o3d.Vector3dVector(cld)
        o3d.geometry.estimate_normals(cloud,
                                     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
        normal = np.asarray(cloud.normals)
        return normal

    def get_point_normal(self, cld):
        cloud = o3d.geometry.PointCloud()
        cld = cld.astype(np.float32)
        cloud.points = o3d.Vector3dVector(cld)
        o3d.geometry.estimate_normals(cloud,
                                     search_param=o3d.geometry.KDTreeSearchParamKNN(50))
        return cloud

    def pcd_down_sample(self, src, voxel_size, n_sample_points, original_idx):
        # # numpy -> PointCloud
        # cloud = o3d.geometry.PointCloud()
        # src = src.astype(np.float32)
        # cloud.points = o3d.Vector3dVector(src)
        original_idx = np.array([original_idx, original_idx, original_idx]).transpose()
        color_idx = original_idx.tolist()
        src.colors = o3d.Vector3dVector(color_idx)

        max_bound = src.get_max_bound()
        min_bound = src.get_min_bound()
        # 截取物体范围内的点
        max_distance = 1.5  # m
        min_distance = 0.3  # m
        src_tree = o3d.KDTreeFlann(src)
        _, max_idx, _ = src_tree.search_radius_vector_3d([0, 0, 0], max_distance)
        _, min_idx, _ = src_tree.search_radius_vector_3d([0, 0, 0], min_distance)
        src = o3d.select_down_sample(src, max_idx)
        if len(min_idx) > 0:
            src_tree = o3d.KDTreeFlann(src)
            _, min_idx, _ = src_tree.search_radius_vector_3d([0, 0, 0], min_distance)
            src = o3d.select_down_sample(src, min_idx, True)

        _, point_idx = o3d.voxel_down_sample_and_trace(
            src, voxel_size=voxel_size,
            min_bound=min_bound, max_bound=max_bound)

        max_idx_in_row = np.max(point_idx, axis=1)
        pcd_down = o3d.select_down_sample(src, max_idx_in_row)
        # pcd_down_tree = o3d.KDTreeFlann(pcd_down)
        # _, idx, _ = pcd_down_tree.search_radius_vector_3d([0,0,0], 1500)
        # pcd_down = o3d.select_down_sample(pcd_down, idx)

        # 如果采样后的点数大于指定值, 随机减少一定数量的
        n_points = len(max_idx_in_row)

        np.random.seed(666)
        if n_points > n_sample_points:
            n_minus = n_points - n_sample_points
            n_pcd_dow = n_points
            minus_idx = np.random.choice(n_pcd_dow, n_minus, replace=False)
            pcd_down = o3d.select_down_sample(pcd_down, minus_idx, True)
            return pcd_down
        elif n_points < n_sample_points:
            n_add = n_sample_points - n_points
            n_cuted_points = len(src.points)
            n_unsample_points = n_cuted_points - max_idx_in_row.shape[0]
            if n_add < n_unsample_points:
                # select unsampled points

                unsample_points = o3d.select_down_sample(src, max_idx_in_row, True)

                add_idx = np.random.choice(n_unsample_points, n_add, replace=False)
                add_points = o3d.select_down_sample(unsample_points, add_idx)
                pcd_down += add_points
                return pcd_down
            else:
                return None

        else:
            return pcd_down






    def get_item(self, item_name):
        words = item_name.split()
        folderName = words[0]
        rgbName = words[1]
        sceneId = rgbName[:-4]
        depthName = words[2]
        segName = words[3]
        depthPath = os.path.join(folderName, "depth/{}".format(depthName))
        rgbPath = os.path.join(folderName, "rgb/{}".format(rgbName))
        segPath = os.path.join(folderName, "mask_visib/{}".format(segName))

        # 读取数据
        with Image.open(depthPath) as di:
            dpt = np.array(di)
        with Image.open(segPath) as li:
            labels = np.array(li)  # labels : mask
            labels = (labels > 0).astype("uint8")  # 转换为8位无符号整型数据,够用吗?
        with Image.open(rgbPath) as ri:
            if self.add_noise:
                ri = self.trans_color(ri)
            rgb = np.array(ri)[:, :, :3]

        # rgb预处理
        rgb = rgb[:, :, ::-1].copy()  # # r b 互换
        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        # cld预处理
        cam_parameter = self.get_cam_parameter(folderName, sceneId)
        K = np.resize(np.array(cam_parameter['cam_K']), (3, 3))
        depth_scale = cam_parameter['depth_scale']
        cam_scale = 1000/depth_scale  # BOP中的深度以0.1mm为单位, 转换成m需要除以10000

        cld, choose = self.bs_utils.dpt_2_cld(dpt, cam_scale, K)  # k:内参, cam_scale: 设置为1.0,不知道什么含义
                                                                  # choose : 深度图中不为0的像素的索引
        # 对choose重新排序
        choose_rerank = np.array([i for i in range(choose.shape[0])])
        cld_normal_o3d = self.get_point_normal(cld)
        cld_normal_down = self.pcd_down_sample(cld_normal_o3d, self.voxel_size, self.n_sample_points, choose_rerank)
        original_idx_down = np.asarray(cld_normal_down.colors).astype(np.int)
        original_idx_down = original_idx_down[:, 0].tolist()
        cld_down = np.asarray(cld_normal_down.points)

        # cld_rgb_normal预处理
        rgb_lst = []
        for ic in range(rgb.shape[0]):
            rgb_lst.append(
                rgb[ic].flatten()[choose].astype(np.float32)
            )  # 提取点云对应的像素
        rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

        cld_rgb = np.concatenate((cld, rgb_pt), axis=1)
        cld_rgb = cld_rgb[original_idx_down, :]

        normal = np.asarray(cld_normal_down.normals)  # 计算法线
        normal[np.isnan(normal)] = 0.0
        cld_rgb_normal = np.concatenate((cld_rgb, normal), axis=1)

        # choose 预处理
        choose = np.array([choose])
        choose_dow = choose[:, original_idx_down]

        # cls_id 预处理 (作用不明)
        cls_ids = np.zeros((2, 1))

        # labels 预处理
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]  # 转成单通道
        labels = labels.flatten()[choose][0]  # labels : mask
        labels = labels[original_idx_down].astype(np.int32)
        n_target = labels.nonzero()
        n_target = len(n_target[0])
        if n_target < 10:
           # print("n_target < 10")
            return None







        # choose: 降采样后的点对应的原深度图上的索引
        return torch.from_numpy(rgb.astype(np.float32)), \
               torch.from_numpy(cld_down.astype(np.float32)), \
               torch.from_numpy(cld_rgb_normal.astype(np.float32)), \
               torch.LongTensor(choose_dow.astype(np.int32)), \
               torch.LongTensor(cls_ids.astype(np.int32)), \
               torch.LongTensor(labels.astype(np.int32))







    def __len__(self):
        return len(self.data_list)

    # 接收一个索引,然后返回用于训练的数据和标签
    def __getitem__(self, idx):  # 调用函数实例时传入
        print("load {}th data...".format(idx))
        item_name = self.data_list[idx]
        data = self.get_item(item_name)
        while data is None:
            print("to few points:{}".format(idx))
            idx = np.random.randint(0, len(self.data_list))
            item_name = self.data_list[idx]
            print("replaced by :{}".format(idx))
            data = self.get_item(item_name)
        return data

def main():
    cls = "5"
    show = True
    test_all = False
    test_range = 10
    pre_processing = False

    # dataListPath = '/media/yumi/Datas/6D_Dataset/BOP_Dataste/LM-O/BOP_test19-20/validList_8.txt'
    dataListPath = './BOP_Dataset/LM-O/train_pbr/trainListSplit100_{}.txt'.format(cls)

    config = Config(dataListPath, dataListPath)

    ds = BOPDataset(dataListPath, cls)
    ds_loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=False,
        num_workers=6
    )
    time_start = time.clock()

    # 没有预处理数据集的情况下,评估数据加载程序
    if test_all:
        for ibs, batch in enumerate(ds_loader):
            data = [item.numpy() for item in batch]
            rgb, pcd_down, cld_rgb_nrm, choose, cls_ids, labels = data

    if test_range:
        for i in range(test_range):
            data = ds.__getitem__(i)
            if data:
                data = [item.numpy() for item in data]
                rgb, pcd_down, cld_rgb_nrm, choose, cls_ids, labels = data
                # n = pcd_down.size()
                # print(n)
                if show:
                    # 显示rgb
                    rgb1 = rgb.transpose(1, 2, 0) / 255
                    rgb1 = rgb1[:, :, ::-1]
                    plt.figure("rgb")
                    plt.imshow(rgb1)
                    plt.title("rgb")
                    plt.show()

                    # 显示法线
                    nrm_map = ds.bs_utils.get_normal_map(cld_rgb_nrm[:, 6:], choose[0])

                    plt.figure("nrm_map")
                    plt.imshow(nrm_map)
                    plt.title("nrm_map")
                    plt.show()

                    # 显示rgb_point_map
                    rgb_point_map = ds.bs_utils.get_rgb_pts_map(
                        cld_rgb_nrm[:, 3:6], choose[0])
                    plt.figure("rgb_point_map")
                    plt.imshow(rgb_point_map)
                    plt.title("rgb_point_map")
                    plt.show()

                    # 显示点云 和 颜色 和 label
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.Vector3dVector(cld_rgb_nrm[:, 0:3])
                    color = cld_rgb_nrm[:, 3:6] / 255
                    # color[:, [0, 2]] = color[:, [2, 0]]
                    pcd.colors = o3d.Vector3dVector(color[:, ::-1])
                    pcd.normals = o3d.Vector3dVector(cld_rgb_nrm[:, 6:])
                    print(pcd)
                    pcd_mask = labels.nonzero()
                    label_idx = pcd_mask[0].tolist()
                    target_pcd = o3d.select_down_sample(pcd, label_idx)
                    target_pcd.paint_uniform_color([0, 255, 0])
                    pcd = o3d.select_down_sample(pcd, label_idx, True)
                    o3d.visualization.draw_geometries([pcd, target_pcd])

                print(i)
            else:
                print("{} data is None".format(i))

    if pre_processing:
        for ibs, batch in enumerate(ds_loader):
            data = [item.numpy() for item in batch]
            rgb, pcd_down, cld_rgb_nrm, choose, cls_ids, labels = data
            #  保存cld_rgb_nrm
            cld_rgb_nrm = o3d.geometry.PointCloud()
            cld_rgb_nrm.points = o3d.Vector3dVector(cld_rgb_nrm[:, 0:3])
            color = cld_rgb_nrm[:, 3:6]
            cld_rgb_nrm.colors = o3d.Vector3dVector(color)
            cld_rgb_nrm.normals = o3d.Vector3dVector(cld_rgb_nrm[:, 6:])




    time_end = time.clock()
    print("time:{}".format(time_end - time_start))










if __name__ == '__main__':
    main()