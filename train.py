import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg




def main(cfg):

    cfg["mode"] = "train"
    
    initFire(cfg)

    model = FireModel(cfg)

    data = FireData(cfg)

    if cfg['show_data']:
        data.showTrainData()
    else:  
        train_loader, val_loader = data.getTrainValDataloader()

        runner = FireRunner(cfg, model)
        runner.train(train_loader, val_loader)


if __name__ == '__main__':
    main(cfg)