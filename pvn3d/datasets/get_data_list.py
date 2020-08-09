import os
import json
import numpy as np

cls_id = 8
trainPath = '~/Project/PVN3D/pvn3d/datasets/BOP/BOP_Dataste/LM-O/train_pbr'
testPath = '~/Project/PVN3D/pvn3d/datasets/BOP/BOP_Dataste/LM-O/BOP_test19-20'
splitTrainList = True
n_splitTrainList = 10000

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
    if splitTrainList:
        with open(trainListPath, 'r') as f:
            data = f.readlines()
            np.random.seed(666)
            np.random.shuffle(data)

            trainPoint = n_splitTrainList

            for i, image in enumerate(data):
                if i < trainPoint:
                    trainListSplit.write(image)


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





