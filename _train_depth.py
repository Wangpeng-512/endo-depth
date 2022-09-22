from tools.trainers.endodepth import EndoDepthTrainer
from tools.options.endodepth import EndoDepthOptions

if __name__ == "__main__":
    options = EndoDepthOptions()
    opt = options.parse()

    opt.model_name = "edp2022"
    opt.data_path = [
        "/data/Datasets/blender/blender-duodenum-5-211126",
        "/data/Datasets/blender/blender-duodenum-5(dark)-211126",
        "/data/Datasets/blender/blender-duodenum-5(light)-211126",
    ]
    opt.val_path = "/data/Datasets/blender/blender-duodenum-3-210909"
    opt.log_dir = "/opt/ytom/edp2022"
    opt.num_epochs = 25
    opt.log_frequency = 1
    opt.save_frequency = 1
    opt.png = True
    opt.learning_rate = 0.0001 # 初始学习率
    opt.lr_decade_coeff = 0.1 # 规划衰减率
    opt.scheduler_step_size = [20] # 规划步骤
    opt.betas = (0.9, 0.999)
    opt.weight_decay = 0.01

    opt.num_layers = 18 # 主干:resnet18
    opt.height = 320
    opt.width = 320
    opt.frame_ids = [0]
    opt.scales = [0] # 输出scale # [0, 1, 2, 3]

    opt.batch_size = 20
    opt.num_workers = 4

    opt.use_depth_loss = False
    opt.use_normal_loss = False
    opt.use_smooth_loss = True

    opt.weight_depth_loss = 1
    opt.weight_normal_pc_loss = 0.01
    opt.weight_normal_norm_loss = 1
    opt.weight_smooth_loss = 0.01

    trainer = EndoDepthTrainer(options=opt, verbose=2)
    # trainer.val()
    trainer.train()
