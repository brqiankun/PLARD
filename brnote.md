[plard_reop地址](https://github.com/brqiankun/PLARD)



# 下载kitti数据集road


挂载数据集
```
sudo mount /dev/sda2 /media/br
```

需要安装google protobuf  cityscapes等库， 手机无法连接外网
plard 在mmseg_plard下测试跑通
```
python test.py --model_path /path/to/plard_kitti_road.pth
python train.py --arch plard --dataset kitti_road --n_epoch 2 --batch_size 1
```
显存不足，训练不了模型  是否需要升级电脑？

input tr_image's size: torch.Size([1, 3, 384, 1280])
input tr_lidar's size: torch.Size([1, 1, 384, 1280])
output's shape: torch.Size([1, 2, 384, 1280])

### POINTS

标注数据: 3D point cloud, image
标注内容:  3D bbox
离线标注: 支持
输出数据格式: 
是否可转化为Nuscenes格式

基于网页web的开源3D点云标注框架
CherryPy  3Dbounding box
前端是WebGL Three.js 在线标注？
有一个docker


1. 数据预处理
2. 可选得使用AI算法包括，det，track，2D&3D fusion 生成初始标注结果
3. 可视化
4. 交互得修改标签，调整2D/3D标注框
5. 扩展单帧结果到多帧

主要是3D Box的标注，但是可以调整3D box在投影角度
支持box fitting和 annotation transfer among frames

输出格式是？
使用kitti做了对比

语义分割多帧融合

1. 图像
2. bev的
3. slam的点云

最终构建两个数据集？ 一个语义，一个3D目标检测？
点云，图像
自动求导
https://pytorch.org/docs/master/notes/autograd.html

修改已有预训练模型
https://blog.csdn.net/andyL_05/article/details/108930240
```
python -m pip install cityscapesscripts
```

