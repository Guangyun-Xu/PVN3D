import torch
from pvn3d.datasets.BOP.BOP_dataset import BOPDataset
from torch.utils.data import DataLoader
from pvn3d.lib.pvn3d import PVN3D
from pvn3d.lib.utils.sync_batchnorm import convert_model
import os
import pickle as pkl
import torch.nn as nn
import tqdm
import numpy as np
import open3d as o3d
__all__ = [o3d]

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):


    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None

def predSeg(model, data, epoch=0, obj_id=''):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = [item.to("cuda", non_blocking=True) for item in data]
        rgb, pcld, cld_rgb_nrm, choose, cls_ids, labels = cu_dt
        pred_rgbd_seg = model(cld_rgb_nrm, rgb, choose)
        _, segIdx = torch.max(pred_rgbd_seg, -1)  # segIdx:每一行中,最大的数在行中的索引
        predMask = segIdx[0]
        objMask = predMask == 1
        predPoints = pcld[:, objMask,:]
        predPoints = predPoints[0]
        predPoints = predPoints.cpu().numpy()
        predMask = objMask.cpu().numpy()
        # target_idx = target_idx == 1

    return predPoints, predMask




def main():

    # 加载数据
    cls_id = '9'
    n_data = 500
    # testDataList = './datasets/BOP/BOP_Dataset/LM-O/train_pbr/trainListSplit{}_{}.txt'.format(n_data, cls_id)
    testDataList = './datasets/BOP/BOP_Dataset/LM-O/BOP_test19-20/validList_{}.txt'.format(cls_id)
    testDataset = BOPDataset(testDataList, cls_id)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=1, shuffle=False,
                                num_workers=6)

    # 定义模型并加载参数
    checkPointPath = './datasets/BOP/BOP_Dataset/LM-O/checkPoint/{}_pvn3d_best_{}.pth.tar'.format(cls_id, n_data)
    model = PVN3D(
        num_classes=2, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=20000
    ).cuda()
    model = convert_model(model)  # 搞清楚什么作用
    model.cuda()
    checkpoint_status = load_checkpoint(
        model, None, filename=checkPointPath
    )

    model = nn.DataParallel(model)

    # 推理
    for i, data in tqdm.tqdm(
        enumerate(testDataLoader), leave=False, desc="val"
    ):
        if data:
            # scene
            scenePointcloud = data[2][0].numpy()
            scenePointcloudO3D = o3d.geometry.PointCloud()
            scenePointcloudO3D.points = o3d.Vector3dVector(scenePointcloud[:, 0:3])
            color = scenePointcloud[:, 3:6]/255
            scenePointcloudO3D.colors = o3d.Vector3dVector(color[:, ::-1])

            # label
            labels = data[5][0].numpy()

            # pre
            predPoints, predPointsMask = predSeg(model, data, epoch=i, obj_id=cls_id)

            # mask
            TP_mask = labels & predPointsMask  # true positive
            FN_mask = labels - TP_mask  # false negative
            FP_mask = predPointsMask - TP_mask  # false positive

            # idx
            TP_idx = TP_mask.nonzero()[0]
            FN_idx = FN_mask.nonzero()[0]
            FP_idx = FP_mask.nonzero()[0]

            # points
            TP_points = o3d.select_down_sample(scenePointcloudO3D, TP_idx)
            FN_points = o3d.select_down_sample(scenePointcloudO3D, FN_idx)
            FP_points = o3d.select_down_sample(scenePointcloudO3D, FP_idx)

            # set color
            TP_points.paint_uniform_color([255, 0, 0])  # red
            FN_points.paint_uniform_color([0, 255, 0])  # green
            FP_points.paint_uniform_color([0, 0, 255])  #

            # remove pred and target from scene
            predPointsIdx = predPointsMask.nonzero()[0]
            target_unpred_and_pred_points_idx = np.append(predPointsIdx, FN_idx)
            scenePointcloudO3D = o3d.select_down_sample(scenePointcloudO3D, target_unpred_and_pred_points_idx, True)

            print(len(TP_idx))

            o3d.visualization.draw_geometries([scenePointcloudO3D, TP_points, FN_points, FP_points])
        else:
            print("{}th data is None!".format(i))


if __name__ == '__main__':
    main()