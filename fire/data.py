
import os
import random
import numpy as np
from sklearn.model_selection import KFold

import cv2
from torchvision import transforms

from fire.datatools import getDataLoader, getFileNames
from fire.dataaug_user import TrainDataAug



class FireData():
    def __init__(self, cfg):
        
        self.cfg = cfg


    def getTrainValDataloader(self):

        print("[INFO] Use kflod to split data: k=%d val_fold=%d" % (self.cfg['k_flod'],self.cfg['val_fold']))
        class_names = ['fake', 'true']#os.listdir(self.cfg['trainval_path'])
        print("Class names:", class_names) #Class names: ['fake', 'true']

        all_data = []
        for data_class in self.cfg['trainval_path']:
            for cid, class_name in enumerate(class_names):

                data_dir = os.path.join(data_class,class_name)
                data_paths = getFileNames(data_dir)
                print(class_name, "count images: ", len(data_paths))
                for data_path in data_paths:
                    all_data.append([data_path, cid])

        print("[INFO] Total trainval:",len(all_data))

        all_data.sort()
        random.shuffle(all_data)

        fold_count = int(len(all_data)/self.cfg['k_flod'])
        val_data = all_data[fold_count*self.cfg['val_fold']:fold_count*(self.cfg['val_fold']+1)]
        train_data = all_data[:fold_count*self.cfg['val_fold']]+all_data[fold_count*(self.cfg['val_fold']+1):]
        print("[INFO] Split train: %d  val: %d" % (len(train_data), len(val_data)))

        input_data = [train_data, val_data]
        train_loader, val_loader = getDataLoader("trainval", 
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader



    def getEvalDataloader(self):
        
        # class_names = os.listdir(self.cfg['eval_path'])
        class_names = ['fake', 'true']
        print("Class names:", class_names)

        all_data = []
        for cid, class_name in enumerate(class_names):

            data_dir = os.path.join(self.cfg['eval_path'],class_name)
            data_paths = getFileNames(data_dir)
            print(class_name, "count images: ", len(data_paths))
            for data_path in data_paths:
                all_data.append([data_path, cid])

        print("[INFO] Total eval:",len(all_data))
        input_data = [all_data]
        data_loader = getDataLoader("eval", 
                                        input_data,
                                        self.cfg)
        return data_loader

    def getTestDataloader(self):
        data_names = getFileNames(self.cfg['test_path'])
        input_data = [data_names]
        data_loader = getDataLoader("test", 
                                    input_data,
                                    self.cfg)
        return data_loader


    def showTrainData(self, show_num = 200):
        #show train data finally to exam

        show_dir = "show_img"
        show_path = os.path.join(self.cfg['save_dir'], show_dir)
        print("[INFO] Showing traing data in ",show_path)
        if not os.path.exists(show_path):
            os.makedirs(show_path)


        img_path_list = getFileNames(self.cfg['train_path'])[:show_num]
        transform = transforms.Compose([TrainDataAug(self.cfg['img_size'])])


        for i,img_path in enumerate(img_path_list):
            #print(i)
            img = cv2.imread(img_path)
            img = transform(img)
            img.save(os.path.join(show_path,os.path.basename(img_path)), quality=100)

    