[plard_reop地址](https://github.com/brqiankun/PLARD)

![plard](./imgs/main.png "plard")

plard 

ADI图像？？？？
表面法线图？？？

结果注意力
中间结果，和中间特征



# 下载kitti数据集road


挂载数据集
```
sudo mount /dev/sda2 /media/br
```

需要安装google protobuf  cityscapes等库， 手机无法连接外网
pip install opencv-python  # 安装cv2
plard 在mmseg_plard下测试跑通
```
python test.py --model_path /path/to/plard_kitti_road.pth
python train.py --arch plard --dataset kitti_road --n_epoch 2 --batch_size 1
```

test 显存占用   3408MiB /  3911MiB

显存不足，训练不了模型  是否需要升级电脑？

input tr_image's size: torch.Size([1, 3, 384, 1280])
input tr_lidar's size: torch.Size([1, 1, 384, 1280])
output's shape: torch.Size([1, 2, 384, 1280])


# DeepScene 越野环境数据集
晚上回去下载越野场景数据集DeepScene


## POINTS

标注数据: 3D point cloud, image
标注内容:  3D bbox 主要用于3D目标检测，对应的数据集kitti
离线标注: 支持
输出数据格式: json格式的label
是否可转化为Nuscenes格式: 不是nuscence格式，类似于kitti

代码地址: 
https://github.com/naurril/SUSTechPOINTS
使用docker可以运行,使用如下命令启动网页后端
```
docker run -it -d -p 8081:8081 --mount type=bind,source=/path/to/data,target=/root/SUSTechPOINTS/data juhaoming/sustechpoints:v1.0.0 /bin/bash
```
#### 数据格式
按照example的格式
点云数据需要是pcd格式，一帧点云(1个xxx.pcd)，对应的图片文件名需要前缀相同(xxx.png)。标注后的输出结果在./label下(xxx.json)
```
+- data
    +- scene1
      +- lidar
            +- 0000.pcd
            +- 0001.pcd
      +- camera
            +- front
                +- 0000.jpg
                +- 0001.jpg
            +- left
                +- ...
      +- aux_lidar
            +- front
                +- 0000.pcd
                +- 0001.pcd
      +- radar
            +- front_points
                +- 0000.pcd
                +- 0001.pcd
            +- front_tracks
                +- ...
      +- calib
            +- camera
                +- front.json
                +- left.json
            +- radar
                +- front_points.json
                +- front_tracks.json
      +- label
            +- 0000.json
            +- 0001.json
    +- scene2
```



标注数据: 3D point cloud, image
标注内容:  3D bbox 主要用于3D目标检测，对应的数据集kitti
离线标注: 支持
输出数据格式: json格式的label
是否可转化为Nuscenes格式: 不是nuscence格式，类似于kitti

基于网页web的开源3D点云标注框架elp
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

### 数据格式
```
+- data
    +- scene1
      +- lidar
            +- 0000.pcd
            +- 0001.pcd
      +- camera
            +- front
                +- 0000.jpg
                +- 0001.jpg
            +- left
                +- ...
      +- aux_lidar
            +- front
                +- 0000.pcd
                +- 0001.pcd
      +- radar
            +- front_points
                +- 0000.pcd
                +- 0001.pcd
            +- front_tracks
                +- ...
      +- calib
            +- camera
                +- front.json
                +- left.json
            +- radar
                +- front_points.json
                +- front_tracks.json
      +- label
            +- 0000.json
            +- 0001.json
    +- scene2
```
按照example的格式
点云数据需要是pcd格式，目前是一帧点云(1个xxx.pcd)，对应的图片文件名需要前缀相同(xxx.png)

自动求导
https://pytorch.org/docs/master/notes/autograd.html

修改已有预训练模型
https://blog.csdn.net/andyL_05/article/details/108930240
```
python -m pip install cityscapesscripts
```

### plard 网络结构
__dropout__ During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.

主要耗时在lidar图形的处理，lidar特征的融合只占到0.1??

plard lidar reshape_872 模型输入出维度变成未知的?? 需要搞清修改模型结构吗


### LoDNN  Lidar_only FCN  Fast LIDAR-based Road Detectio Using Fully Convolutional Neural Networks
仅使用激光雷达数据进行道路检测。
在点云顶视图进行检测
“Multi-view 3d object detection network for autonomous driving”  生成俯视图像

在点云的X-Y平面上创建网格，将每个点云分配到其中一个单元。x:[6, 46], y:[-10, 10].  根据kitti的标准，每个格的大小为0.1 * 0.1(meters)
然后为每个网格计算一些数据: 点数，反射率，标准差，最大/最小高度。将网格单元设置为像素, 生成6幅包含上述统计数据的图像。 俯视图分辨率是(200, 400)
该模型的输出图像是包含尺度信息的

数据增强使用了旋转图像的方式。绕激光雷达Z轴旋转[-30, 30]度

project the point cloud int othe corresponding camera-view annotation in order to determine which of its points belong to the road.
将点云投影到对应的相机标签图中，来确定哪些点属于道路。之后使用前述方式，将带点云转换成俯视图像，最终得到俯视视角的注释。

## BEV 感知：
相机BEV转3D:
1. IPM，假设平坦平面
2. Lift-Splat-Shoot (LSS)  bevfusion, bevdepth, 使用了GPU Kernel加速深度估计，耗费显存较大
3. MLP 不太行
4. transformer 数据依赖，训练困难，实车部署困难


### LSS 
https://github.com/nv-tlabs/lift-splat-shoot
[LSS代码解读](https://zhuanlan.zhihu.com/p/567880155)

### semantic-kitt
http://semantic-kitti.org/resources.html#devkit


**TODO**
```
torch.cuda.empty_cache()  # 清理cuda缓存
warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")  更新API
```
1. 需要修改plard的模型
```
torch.cuda.empty_cache()  # 清理cuda缓存
warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")  更新API
```
2. 为何推理不足3g, 而训练8g都不够, 或者更改模型，直接使用纯图像的方案. 删除lidar部分的分支。检测性能是否变化

3. 剪枝，蒸馏？