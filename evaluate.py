import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
from glob import glob



def main(cfg):
    data_path = [
        # "../data/gan_yellow",
        "../data/deepfake_dfmnist",
        "../data/gan_wanghong", 
        "../data/deepfake_celeb", 
        "../data/forgery/data"
        ]
    
    cfg["cfg_verbose"] = False
    cfg["GPU_ID"] = "0"
    cfg["mode"] = "eval"
    folder_name = "20230831-084046-mobilevitv2_100"
    cfg["model_name"] = folder_name.split("-")[-1]
    cfg["model_path"] = glob('output/{}/*_e*.pth'.format(folder_name))[0]
    
    for eval_path in data_path:
        cfg["eval_path"] = eval_path
        
        print("{} eval_path : {} {}".format("-"*20, eval_path, "-"*20))
        
        initFire(cfg)

        model = FireModel(cfg)
        data = FireData(cfg)
        # data.showTrainData()
        
        train_loader = data.getEvalDataloader()
        runner = FireRunner(cfg, model)
        runner.modelLoad(cfg['model_path'])
        runner.evaluate(train_loader)

if __name__ == '__main__':
    main(cfg)