import os
import json
import numpy as np
from pvn3d.common_BOP import Config


config = Config()
trainPath = './BOP_Dataset/LM-O/train_pbr'
testPath = './BOP_Dataset/LM-O/BOP_test19-20'
splitTrainList = True
n_splitTrainList = 500
obj_list = config.LMO_obj_list

if obj_list[0]:
    for obj_id in obj_list:
        cls_id = obj_id
        dataListPath = os.path.join(trainPath, 'dataList_{}.txt'.format(cls_id))
        trainListPath = os.path.join(trainPath, 'trainList_{}.txt'.format(cls_id))
        trainListSplitPath = os.path.join(trainPath, 'trainListSplit{}_{}.txt'.format(n_splitTrainList, cls_id))
        validListPath = os.path.join(testPath, 'validList_{}.txt'.format(cls_id))
        dataList = open(dataListPath, 'w')
        trainList = open(trainListPath, 'w')
        trainListSplit = open(trainListSplitPath, 'w')
        validList = open(validListPath, 'w')

        if trainPath == testPath:
            files = os.listdir(trainPath)
            for subFile in files:
                if os.path.isdir(os.path.join(trainPath, subFile)):
                    labelPath = os.path.join(trainPath, subFile, 'scene_gt.json')
                    with open(labelPath, 'r') as f:
                        data = json.load(f)
                        for i in data:
                            sceneKey = str(i)
                            scene = data[sceneKey]
                            for j, obj in enumerate(scene):
                                objId = obj['obj_id']
                                if objId == cls_id:
                                    objMaskId = "{:0>6d}".format(int(j))
                                    sceneId = "{:0>6d}".format(int(i))
                                    folderName = os.path.join(trainPath, subFile)
                                    scenePath = "{} {}.jpg {}.png {}_{}.png".format(
                                        folderName, sceneId, sceneId, sceneId, objMaskId)
                                    # print(scenePath)
                                    lineScene = scenePath + '\n'
                                    dataList.write(lineScene)
            dataList.close()

            with open(dataListPath, 'r') as f:
                data = f.readlines()
                np.random.seed(666)
                np.random.shuffle(data)
                number = len(data)

                trainPoint = int(number * 0.8)

                for i, image in enumerate(data):
                    if i < trainPoint:
                        trainList.write(image)
                    else:
                        validList.write(image)
            trainList.close()
            validList.close()

        else:
            trainFiles = os.listdir(trainPath)
            for subFile in trainFiles:
                if os.path.isdir(os.path.join(trainPath, subFile)):
                    labelPath = os.path.join(trainPath, subFile, 'scene_gt.json')
                    with open(labelPath, 'r') as f:
                        data = json.load(f)
                        for i in data:
                            sceneKey = str(i)
                            scene = data[sceneKey]
                            for j, obj in enumerate(scene):
                                objId = obj['obj_id']
                                if objId == cls_id:
                                    objMaskId = "{:0>6d}".format(int(j))
                                    sceneId = "{:0>6d}".format(int(i))
                                    folderName = os.path.join(trainPath, subFile)
                                    scenePath = "{} {}.jpg {}.png {}_{}.png".format(
                                        folderName, sceneId, sceneId, sceneId, objMaskId)
                                    # print(scenePath)
                                    lineScene = scenePath + '\n'
                                    trainList.write(lineScene)
            trainList.close()
            if splitTrainList:
                print("list obj:{} in {}...... ".format(cls_id, n_splitTrainList))
                with open(trainListPath, 'r') as f:
                    data = f.readlines()
                    np.random.seed(666)
                    # np.random.shuffle(data)

                    trainPoint = n_splitTrainList

                    for i, image in enumerate(data):
                        if i < trainPoint:
                            trainListSplit.write(image)
            trainListSplit.close()  # 必须加上close, 否者数据容易丢失

            testFiles = os.listdir(testPath)
            for subFile in testFiles:
                if os.path.isdir(os.path.join(testPath, subFile)):
                    labelPath = os.path.join(testPath, subFile, 'scene_gt.json')
                    with open(labelPath, 'r') as f:
                        data = json.load(f)
                        for i in data:
                            sceneKey = str(i)
                            scene = data[sceneKey]
                            for j, obj in enumerate(scene):
                                objId = obj['obj_id']
                                if objId == cls_id:
                                    objMaskId = "{:0>6d}".format(int(j))
                                    sceneId = "{:0>6d}".format(int(i))
                                    folderName = os.path.join(testPath, subFile)
                                    scenePath = "{} {}.png {}.png {}_{}.png".format(
                                        folderName, sceneId, sceneId, sceneId, objMaskId)
                                    # print(scenePath)
                                    lineScene = scenePath + '\n'
                                    validList.write(lineScene)
            validList.close()








