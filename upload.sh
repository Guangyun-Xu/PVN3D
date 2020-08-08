#!/bin/bash
# 变量名和等号之间不能有空格
seversPath=/home/casia_robot_mind/Project
localPath=/home/yumi/Project/6D_pose_estmation/PVN3D
password="casia_robot_mind_123456"
if [[ $seversPath =~ "." ]]
then
  sshpass -p $password rsync $localPath \
  casia_robot_mind@172.18.74.33:$seversPath
else
  sshpass -p $password rsync -r -l $localPath \
  casia_robot_mind@172.18.74.33:$seversPath
fi