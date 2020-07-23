#!/usr/bin/env python3
import os

import cv2

import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pvn3d.common import Config
import pickle as pkl
from pvn3d.lib.utils.basic_utils import Basic_Utils
import yaml
from cv2 import imshow, waitKey
import json

# import open3d as o3d
# __all__ = [o3d]


DEBUG = False


class LM_O_Dataset():

    def __init__(self,  data_list_path, cls_id):

        # self.config = Config(dataset_name='linemod', cls_type=cls_type)
        self.bs_utils = Basic_Utils()
        self.n_sample_points = 10000

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        # self.obj_dict = self.config.lm_obj_dict

        self.cls_id = cls_id
        # self.cls_type = cls_type
        # self.cls_id = self.obj_dict[cls_type]
        print("load {} data by LM_O_Dataset.py".format(self.cls_id))
        # self.root = os.path.join(self.config.lm_root, 'Linemod_preprocessed')
        # self.cls_root = os.path.join(self.root, "data/%02d/" % self.cls_id)
        self.rng = np.random
        # meta_file = open(os.path.join(self.cls_root, 'gt.yml'), "r")
        # self.meta_lst = yaml.load(meta_file)
        self.dataList = self.bs_utils.read_lines(data_list_path)  # list



        # if dataset_name == 'train':
        #     self.add_noise = True
        #     real_img_pth = os.path.join(
        #         self.cls_root, "train.txt"
        #     )
        #     self.real_lst = self.bs_utils.read_lines(real_img_pth)
        #
        #     rnd_img_pth = os.path.join(
        #         self.root, "renders/{}/file_list.txt".format(cls_type)
        #     )
        #     self.rnd_lst = self.bs_utils.read_lines(rnd_img_pth)
        #
        #     fuse_img_pth = os.path.join(
        #         self.root, "fuse/{}/file_list.txt".format(cls_type)
        #     )
        #     try:
        #         self.fuse_lst = self.bs_utils.read_lines(fuse_img_pth)
        #     except:  # no fuse dataset
        #         self.fuse_lst = self.rnd_lst
        #     self.all_lst = self.real_lst + self.rnd_lst + self.fuse_lst
        # else:
        #     self.add_noise = False
        #     self.pp_data = None
        #     if os.path.exists(self.config.preprocessed_testset_pth) and self.config.use_preprocess:
        #         print('Loading valtestset.')
        #         with open(self.config.preprocessed_testset_pth, 'rb') as f:
        #             self.pp_data = pkl.load(f)
        #         self.all_lst = [i for i in range(len(self.pp_data))]
        #         print('Finish loading valtestset.')
        #     else:
        #         tst_img_pth = os.path.join(
        #             self.cls_root, "test.txt"
        #         )
        #         self.tst_lst = self.bs_utils.read_lines(tst_img_pth)
        #         self.all_lst = self.tst_lst
        print("{}_dataset_size: ".format(data_list_path), len(self.dataList))

    def get_meta_data(self, folderPath, sceneId, obj_id):
        metaFilePath = os.path.join(folderPath, "scene_gt.json")
        with open(metaFilePath, 'r') as f1:
            mateDate = json.load(f1)
            sceneId = sceneId.lstrip('0')
            if sceneId == '':
                sceneId = '0'
            sceneMateDate = mateDate[sceneId]
            for objMeta in sceneMateDate:
                if objMeta['obj_id'] == int(obj_id):
                    return objMeta

    def get_cam_parameter(self, folderPath, sceneId):
        sceneInfoPath = os.path.join(folderPath, "scene_camera.json")
        with open(sceneInfoPath, 'r') as f2:
            sceneInfo = json.load(f2)
            sceneId = sceneId.lstrip('0')
            if sceneId == '':
                sceneId = '0'
            sceneDate = sceneInfo[sceneId]
            return sceneDate



    def real_syn_gen(self, real_ratio=1.0):
        if self.rng.rand() < real_ratio:  # self.rng = np.random
            n_imgs = len(self.real_lst)  # 真实数据的数量
            idx = self.rng.randint(0, n_imgs)  # 将idx设置为0-n_imgs之间的整数
            pth = self.real_lst[idx]
            return pth
        else:
            fuse_ratio = 0.4
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        idx = self.rng.randint(0, len(self.real_lst))
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand() * (hi - lo) + lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1 - 0.25, 1 + .25)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1 - .15, 1 + .15)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        return np.clip(img, 0, 255).astype(np.uint8)

    # def get_normal(self, cld):  # 改成Open3D的
    #     cloud = pcl.
    #     cld = cld.astype(np.float32)
    #     cloud.from_array(cld)
    #     ne = cloud.make_NormalEstimation()
    #     kdtree = cloud.make_kdtree()
    #     ne.set_SearchMethod(kdtree)
    #     ne.set_KSearch(50)
    #     n = ne.compute()
    #     n = n.to_array()
    #     return n

    def get_normal(self, cld):
        cldShape = cld.shape
        normal = np.random.random(cldShape)
        return normal
        # cloud = o3d.geometry.PointCloud()
        # cld = cld.astype(np.float32)
        # cloud.points = o3d.utility.Vector3dVector(cld)
        # o3d.geometry.estimate_normals(cloud,
        #                              search_param=o3d.geometry.KDTreeSearchParamKNN(50))
        # print(cloud.normals)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.cls_root, "depth/{}.png".format(real_item))) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.cls_root, "mask/{}.png".format(real_item))) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        if len(bk_label.shape) < 3:
            bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        else:
            bk_label_3c = bk_label
            bk_label = bk_label[:, :, 0]
        with Image.open(os.path.join(self.cls_root, "rgb/{}.png".format(real_item))) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
            back = back[:, :, ::-1].copy()
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        imshow("msk_back", msk_back)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
              dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def get_item(self, item_name):
        words = item_name.split()
        folderName = words[0]
        rgbName = depthName = sceneId = words[1]
        segName = words[2]
        depthPath = os.path.join(folderName, "depth/{}.png".format(depthName))
        rgbPath = os.path.join(folderName, "rgb/{}.jpg".format(rgbName))
        segPath = os.path.join(folderName, "mask_visib/{}.png".format(segName))

        with Image.open(depthPath) as di:
            dpt = np.array(di)
        with Image.open(segPath) as li:
            labels = np.array(li)  # labels : mask
            labels = (labels > 0).astype("uint8")
        with Image.open(rgbPath) as ri:
            # if self.add_noise:
            #     ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]

        meta = self.get_meta_data(folderName, sceneId, self.cls_id)  # meta 指的是目标物体的位姿和bbox

        R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        T = np.array(meta['cam_t_m2c']) / 1000.0  # 以m为单位
        RT = np.concatenate((R, T[:, None]), axis=1)
        rnd_typ = 'real'
        camParameter = self.get_cam_parameter(folderName, sceneId)
        K = np.resize(np.array(camParameter['cam_K']), (3, 3))
        cam_scale = 10000.0  # BOP中的深度以0.1mm为单位, 转换成m需要除以10000

        rgb = rgb[:, :, ::-1].copy()  # # r b 互换
        msk_dp = dpt > 1e-6
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]  # 转成单通道
        rgb_labels = labels.copy()

        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw
        cld, choose = self.bs_utils.dpt_2_cld(dpt, cam_scale, K)  # k:内参, cam_scale: 设置为1.0,不知道什么含义
        # choose : 深度图中不为0的像素的索引

        labels = labels.flatten()[choose]  # labels : mask
        rgb_lst = []
        for ic in range(rgb.shape[0]):
            rgb_lst.append(
                rgb[ic].flatten()[choose].astype(np.float32)
            )
        rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

        choose = np.array([choose])
        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        if len(choose_2) < 400:  # 如果场景中点云的数量过少,返回None
            return None
        if len(choose_2) > self.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]  # c_mask: 随机的0 1 组成的数组,choose_2:用于降采样
        else:
            choose_2 = np.pad(choose_2, (0, self.n_sample_points - len(choose_2)), 'wrap')

        cld_rgb = np.concatenate((cld, rgb_pt), axis=1)
        cld_rgb = cld_rgb[choose_2, :]
        cld = cld[choose_2, :]

        normal = self.get_normal(cld)[:, :3]
        normal[np.isnan(normal)] = 0.0

        cld_rgb_nrm = np.concatenate((cld_rgb, normal), axis=1)
        choose = choose[:, choose_2]  # 降采样后的像素对应的原图上的索引
        labels = labels[choose_2].astype(np.int32)

        RTs = np.zeros((2, 3, 4))
        kp3ds = np.zeros((2, 5, 3))
        ctr3ds = np.zeros((2, 3))
        cls_ids = np.zeros((2, 1))
        kp_targ_ofst = np.zeros((10000, 5, 3))
        ctr_targ_ofst = np.zeros((10000, 3))
        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            # ctr = self.bs_utils.get_ctr(self.cls_type, ds_type="linemod")[:, None]
            # ctr = np.dot(ctr.T, r.T) + t
            # ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0 * ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1])

            key_kpts = ''
            # if self.config.n_keypoints == 8:
            #     kp_type = 'farthest'
            # else:
            #     kp_type = 'farthest{}'.format(self.config.n_keypoints)
            # kps = self.bs_utils.get_kps(
            #     self.cls_type, kp_type=kp_type, ds_type='linemod'
            # )
            # kps = np.dot(kps, r.T) + t
            # kp3ds[i] = kps

            target = []
            # for kp in kps:
            #     target.append(np.add(cld, -1.0 * kp))
            # target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            # kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        # rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, cls_ids, RTs, labels, kp_3ds, ctr_3ds
        if DEBUG:
            return torch.from_numpy(rgb.astype(np.float32)), \
                   torch.from_numpy(cld.astype(np.float32)), \
                   torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
                   torch.LongTensor(choose.astype(np.int32)), \
                   torch.from_numpy(kp_targ_ofst.astype(np.float32)), \
                   torch.from_numpy(ctr_targ_ofst.astype(np.float32)), \
                   torch.LongTensor(cls_ids.astype(np.int32)), \
                   torch.from_numpy(RTs.astype(np.float32)), \
                   torch.LongTensor(labels.astype(np.int32)), \
                   torch.from_numpy(kp3ds.astype(np.float32)), \
                   torch.from_numpy(ctr3ds.astype(np.float32)), \
                   torch.from_numpy(K.astype(np.float32)), \
                   torch.from_numpy(np.array(cam_scale).astype(np.float32))

        # choose: 降采样后的点对应的原深度图上的索引
        return torch.from_numpy(rgb.astype(np.float32)), \
               torch.from_numpy(cld.astype(np.float32)), \
               torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               torch.LongTensor(cls_ids.astype(np.int32)), \
               torch.LongTensor(labels.astype(np.int32)), \


    def __len__(self):
        return len(self.dataList)

    # 接收一个索引,然后返回用于训练的数据和标签
    def __getitem__(self, idx):  # 调用函数实例时传入
        item_name = self.dataList[idx]
        data = self.get_item(item_name)
        return data



def main():
    # self.config.mini_batch_size = 1
    global DEBUG
    cls = "ape"  # 1
    DEBUG = True
    dataListPath = '/media/yumi/Datas/6D_Dataset/BOP_Dataste/lm-o/train_pbr/trainList_1.txt'

    ds = LM_O_Dataset(dataListPath, cls)
    idx = dict(
        train=0,
        val=0,
        test=0
    )  # {'train': 0, 'val': 0, 'test': 0}

    while True:
        for i in range(5):
            datum = ds.__getitem__(i)  # get_item返回的数据
            bs_utils = ds.bs_utils

            datum = [item.numpy() for item in datum]
            rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, \
            ctr_targ_ofst, cls_ids, RTs, labels, kp3ds, \
            ctr3ds, K, cam_scale = datum
            nrm_map = bs_utils.get_normal_map(cld_rgb_nrm[:, 6:], choose[0])
            imshow('nrm_map', nrm_map)
            rgb = rgb.transpose(1, 2, 0)  # [...,::-1].copy()
            for i in range(22):
                p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
                # rgb = self.bs_utils.draw_p2ds(rgb, p2ds)
                kp3d = kp3ds[i]
                if kp3d.sum() < 1e-6:
                    break
                kp_2ds = bs_utils.project_p3d(kp3d, cam_scale, K)
                rgb = bs_utils.draw_p2ds(
                    rgb, kp_2ds, 3, (0, 0, 255)  # bs_utils.get_label_color(cls_ids[i], mode=1)
                )
                ctr3d = ctr3ds[i]
                ctr_2ds = bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
                rgb = bs_utils.draw_p2ds(
                    rgb, ctr_2ds, 4, (255, 0, 0)  # bs_utils.get_label_color(cls_ids[i], mode=1)
                )
            imshow('{}_rgb'.format(cls), rgb)
            cmd = waitKey(0)
            if cmd == ord('q'):
                exit()
            else:
                continue


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
