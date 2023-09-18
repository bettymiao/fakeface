# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "mobilevitv2_100", 
    "use_timm": True, 
    # swin_tiny_patch4_window7_224 swin_small_patch4_window7_224 swin_base_patch4_window7_224
    # vit_tiny_patch16_224 mobilevitv2_100
    # shufflenetv2 adv-efficientnet-b2 se_resnext50_32x4d  xception resnet18
    'GPU_ID': '2',
    "class_number": 2,

    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":8,

    ### Train Setting
    # 数据集对应：
    # val = gan_yellow + deepfake_dfmnist
    # DeepFakeMnist+ = deepfake_dfmnist
    # Internet celebrity = gan_wanghong
    # Celeb-DF = deepfake_celeb
    # ForgeryNet = forgery
    'trainval_path':[
        "../data/gan_yellow",
        "../data/deepfake_dfmnist"
        ],
    # 'pretrained':'weights/resnet18-5c106cde.pth', #path or ''

    'save_best_only': True,  #only save model if better than before
    'save_one_only': True,    #only save one best model (will del model before)
    "save_dir": "output/",
    'metrics': ['acc'], # default is acc,  can add F1  ...
    # NewOneCenterLoss = fixed one centerloss
    # OneCenter = OneCenter CenterLoss=CenterLoss
    "loss": 'OneCenter', # CE, CEV2-0.5, Focalloss-1 CenterLoss-1 OneCenter...
    "center": 1.0,
    "foc_weight": 1,
    "ad_loss": True,
    
    'show_heatmap':False,
    'show_data':False,

    ### Train Hyperparameters
    "img_size": [224, 224], # [h, w]
    # 'learning_rate':0.001,
    'learning_rate': 1e-4,
    'batch_size':256,
    'epochs':30,
    'optimizer':'AdamW',  #Adam  SGD AdaBelief Ranger
    # SGDR-5-2 SGDR-30-1
    'scheduler':'SGDR-30-1', #default default-0.1-3 SGDR-5-2   step-4-0.8

    'warmup_epoch':0, # 
    'weight_decay' : 0,#0.0001,
    "k_flod":5,
    'val_fold':0,
    "use_early_stop": True,
    'early_stop_patient':30,

    'use_distill':0,
    'label_smooth':0,
    # 'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,

    'dropout':0.5, #before last_linear

    'mixup':False,
    'cutmix':False,
    'sample_weights':None,

    ### Test
    "mode": None,
    'model_path': None, #test model

    'eval_path':"../data/gan_wanghong",#test with label,get test acc
    #gan_yellow gan_wanghong deepfake_dfmnist deepfake_celeb   forgery
    'test_path':"../data/test",#test without label, just show img result
    
    'TTA':False,
    'merge':False,
    'test_batch_size': 1,
    

}
