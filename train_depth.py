'''
Author: Wangpeng-512 1119785034@qq.com
Date: 2022-11-04 09:46:17
LastEditors: Wangpeng-512 1119785034@qq.com
LastEditTime: 2022-11-04 23:28:32
FilePath: \code2\train_depth.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BDot
'''
from tools.trainers.endodepth import plEndoDepth
from tools.options.endodepth import EndoDepthOptions
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    options = EndoDepthOptions()
    opt = options.parse()
    
    opt.backbone = "transformer" # default 'transformer' or 'resnet'
    opt.model_name = "edp2022"
    opt.data_path = "D:\\VScode\\code2\\data\\blender"
    opt.val_path = None  # "/data/Datasets/blender/blender-duodenum-3-210909"
    opt.log_dir = "D:\\VScode\\code2\\data\\edp2022"
    opt.num_epochs = 50
    opt.log_frequency = 1
    opt.save_frequency = 1
    opt.png = True
    opt.learning_rate = 0.01  # 初始学习率
    opt.lr_decade_coeff = 0.88  # 规划衰减率 (~0.001 for 50)
    opt.betas = (0.9, 0.999)
    opt.weight_decay = 0.01

    opt.num_layers = 18  # 主干:resnet18
    opt.height = 320
    opt.width = 320
    opt.frame_ids = [0]
    opt.scales = [0]  # 输出scale # [0, 1, 2, 3]

    opt.batch_size = 5
    opt.num_workers = 1

    opt.use_depth_loss = True
    opt.use_smooth_loss = True
    opt.use_normal_loss = False

    opt.weight_depth_loss = 1
    opt.weight_smooth_loss = 0.5
    opt.weight_normal_pc_loss = 0.25
    opt.weight_normal_norm_loss = 0.25

    opt.test = False

    model = plEndoDepth(options=opt, verbose=2)

    train_loader = DataLoader(
        model.train_set, opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False
    )
    # val_loader = DataLoader(
    #     model.val_set, opt.batch_size, shuffle=False,
    #     num_workers=opt.num_workers, pin_memory=True, drop_last=False
    # )

    checkpoint = ModelCheckpoint(monitor="train_loss")
    early_stop = EarlyStopping(monitor="train_loss",
                               min_delta=1e-8, patience=5, mode="min",
                               stopping_threshold=1e-4, divergence_threshold=10, verbose=False)
    # trainer = pl.Trainer(gpus=1, max_epochs=opt.num_epochs,
    trainer = pl.Trainer(max_epochs=opt.num_epochs,
                        #  fast_dev_run=True,
                         precision=32,
                         limit_train_batches=0.2,
                         callbacks=[checkpoint, early_stop])
    trainer.fit(model, train_loader)
