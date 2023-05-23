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


