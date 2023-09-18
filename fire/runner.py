import time
import gc
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import json

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm
from fire.runnertools import getSchedu, getOptimizer, getLossFunc
from fire.runnertools import clipGradient
from fire.metrics import getF1, AverageMeter
from fire.scheduler import GradualWarmupScheduler
from fire.utils import printDash
from fire.loss import CrossEntropyLoss, CrossEntropyLossOneHot
from datetime import datetime, timedelta
from timm.utils import get_outdir
from sklearn.metrics import accuracy_score, f1_score

class FeatureExtractor():
    #https://github.com/jacobgil/pytorch-grad-cam
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # print(self.model._modules.items())
        for name, module in self.model._modules.items():
            # print(name, module)
            x = module(x)
            if name in self.target_layers:
                # print(name, module, '111')
                x.register_hook(self.save_gradient)
                outputs += [x]
        # b
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        print(len(self.model.features._modules.items()))
        for name, module in self.model.features._modules.items():
            if module == self.feature_module:
                print(name, module)
                target_activations, x = self.feature_extractor(x)
            else:
                x = module(x)

            
        
        x = x.mean(3).mean(2)        #best 99919
        x = self.model.classifier(x)
        #bself.pretrain_model self.features self.classifier

        return target_activations, x


class FireRunner():
    def __init__(self, cfg, model):
        self.cfg = cfg
        if self.cfg['GPU_ID'] != '' :
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()
        ############################################################

        # loss
        self.loss_func = getLossFunc(self.device, cfg)
        self.ce_loss = CrossEntropyLossOneHot().to(self.device)
        # optimizer
        self.optimizer = getOptimizer(self.cfg['optimizer'], 
                                    self.model, 
                                    self.cfg['learning_rate'], 
                                    self.cfg['weight_decay'])
        
        
        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)
        
        if self.cfg['warmup_epoch']:
            self.scheduler = GradualWarmupScheduler(self.optimizer, 
                                                multiplier=1, 
                                                total_epoch=self.cfg['warmup_epoch'], 
                                                after_scheduler=self.scheduler)

        if self.cfg['show_heatmap']:
            self.extractor = ModelOutputs(self.model, self.model.features[12], ['0'])

        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            self.cfg["model_name"],
        ])
        
        if self.cfg["mode"] == "train":
            self.cfg['save_dir'] = get_outdir(self.cfg['save_dir'], exp_name)
            with open(os.path.join(self.cfg['save_dir'], "config.json"), "w") as jf:
                json.dump(self.cfg, jf, indent=2)

    def freezeBeforeLinear(self, epoch, freeze_epochs = 2):
        if epoch<freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif epoch==freeze_epochs:
            for child in list(self.model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = True
        #b

    def train(self, train_loader, val_loader):
        self.onTrainStart()
        for epoch in range(self.cfg['epochs']):
            self.freezeBeforeLinear(epoch, self.cfg['freeze_nonlinear_epoch'])

            self.onTrainStep(train_loader, epoch)
            #self.onTrainEpochEnd()
            self.onValidation(val_loader, epoch)

            if self.earlystop:
                break
        
        self.onTrainEnd()

    def predictRaw(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data).double()

                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    res_dict[os.path.basename(img_names[i])] = pred_score[i].cpu().numpy()

        # pres = np.array(pres)

        return res_dict

    def predict(self, data_loader):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data).double()


                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    res_dict[os.path.basename(img_names[i])] = pred[i].item()

        # pres = np.array(pres)

        return res_dict

    def locate(self, data_loader):
        self.model.eval()
        res_dict = {}
        count = 1
        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in data_loader:
                print("\r",count,'/',len(data_loader),end="",flush=True)
                count+=1
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data)

                for i in range(len(data)):
                    img_name = os.path.basename(img_names[i])
                    label = target.max(1, keepdim=True)[1][i].cpu().numpy().tolist()[0]
                    locate = output[1][i].cpu().numpy().tolist()
                    #print(img_name, label, locate)

                    res_dict[img_name] = [label, int(locate[0]*1000)/1000.0, int(locate[1]*1000)/1000.0]

        # pres = np.array(pres)

        return res_dict

    def evaluate(self, data_loader):
        self.model.eval()

        val_acc_m = AverageMeter()
        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data)

                # target = torch.argmax(target,axis=1)
                # dist = torch.abs(torch.tanh(torch.sum(torch.pow(output[0]-0,2),axis=1)))
                # pred = torch.where(dist >= 0.5, torch.zeros_like(dist), torch.ones_like(dist))

                pred_score = nn.Softmax(dim=1)(output[0])
                if len(target.shape)>1:
                    target = target.max(1, keepdim=True)[1] 
                pred = pred_score.max(1, keepdim=True)[1] # get the index of the max log-probability

                # 计算accuracy
                val_acc = accuracy_score(y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy())
                val_acc_m.update(val_acc)
                
                # batch_pred_score = pred.data.cpu().numpy().tolist()
                # batch_label_score = target.data.cpu().numpy().tolist()
                # print(batch_pred_score)
                # print(batch_label_score)
                # b
                # pres.extend(batch_pred_score)
                # labels.extend(batch_label_score)

        # pres = np.array(pres)
        # labels = np.array(labels)
        #print(pres.shape, labels.shape)

        print('[Info] acc: {:.3f}% \n'.format(100. * val_acc_m.avg))

        # if 'F1' in self.cfg['metrics']:
        #     precision, recall, f1_score = getF1(pres, labels)
        #     print('      precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}\n'.format(
        #           precision, recall, f1_score))

    def onTrainStart(self):
        self.early_stop_value = -1
        self.early_stop_dist = 0
        self.last_save_path = None

        self.earlystop = False
        self.best_epoch = 0

        # log
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))

    def onTrainStep(self, train_loader, epoch):
        self.model.train()

        batch_time = 0
        total_loss_m, center_loss_m, ad_loss_m = AverageMeter(), AverageMeter(), AverageMeter()
        train_acc_m = AverageMeter()
        for batch_idx, (data, target, img_names) in enumerate(train_loader):
            one_batch_time_start = time.time()

            target = target.to(self.device)
            data = data.to(self.device)
            
            with torch.cuda.amp.autocast():
                output = self.model(data)
                if 'NewOneCenterLoss' in self.cfg['loss']:
                   center_loss = self.loss_func(output[0], target, output[1], center=torch.tensor(self.cfg['center']))
                else:
                    center_loss = self.loss_func(output[0], target, output[1])
                if self.cfg["ad_loss"]:
                    ad_loss = self.ce_loss(output[-1], target)
            
            if self.cfg["ad_loss"]:
                loss = center_loss + ad_loss
                ad_loss_m.update(ad_loss, data.size(0))
            else:
                loss = center_loss
            
            total_loss_m.update(loss, data.size(0))
            center_loss_m.update(center_loss, data.size(0))
        
            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])

            self.optimizer.zero_grad()#把梯度置零
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            pred_score = nn.Softmax(dim=1)(output[0])
            if len(target.shape)>1:
                target = target.max(1, keepdim=True)[1] 
            pred = pred_score.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            # 计算accuracy
            train_acc = accuracy_score(y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy())
            train_acc_m.update(train_acc)
            
            one_batch_time = time.time() - one_batch_time_start
            batch_time+=one_batch_time
            # print(batch_time/(batch_idx+1), len(train_loader), batch_idx, 
            #     int(one_batch_time*(len(train_loader)-batch_idx)))
            eta = int((batch_time/(batch_idx+1))*(len(train_loader)-batch_idx-1))

            print_epoch = ''.join([' ']*(4-len(str(epoch+1))))+str(epoch+1)
            print_epoch_total = str(self.cfg['epochs'])+''.join([' ']*(4-len(str(self.cfg['epochs']))))

            log_interval = 10
            if batch_idx % log_interval== 0:
                print('\r',
                    '{}/{} [{}/{} ({:.0f}%)] - ETA: {}, loss: {:.4f}, center_loss: {:.4f}, ad_loss: {:.4f}, acc: {:.4f}  LR: {:f}'.format(
                    print_epoch, print_epoch_total, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    timedelta(seconds=eta),
                    total_loss_m.avg, center_loss_m.avg, ad_loss_m.avg, train_acc_m.avg,
                    self.optimizer.param_groups[0]["lr"]), 
                    end="",flush=True)

    def onTrainEnd(self):
        save_name = 'last_g%s.pth' % (self.cfg['GPU_ID'])
        self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
        self.modelSave(self.last_save_path)
        
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()


    def onValidation(self, val_loader, epoch):
        self.model.eval()
                
        val_loss_m, center_loss_m, ad_loss_m = AverageMeter(), AverageMeter(), AverageMeter()
        val_acc_m = AverageMeter()
        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    # self.val_loss += self.loss_func(output[0], target).item() # sum up batch loss
                    # self.val_loss += self.loss_func(output[0], target, output[1]).item()
                    if 'NewOneCenterLoss' in self.cfg['loss']:
                        center_loss = self.loss_func(output[0], target, output[1], center=torch.tensor(self.cfg['center']))
                    else:
                        center_loss = self.loss_func(output[0], target, output[1])
                    if self.cfg["ad_loss"]:
                        # AD loss
                       ad_loss = self.ce_loss(output[-1], target)
                
                if self.cfg["ad_loss"]:
                    loss = center_loss + ad_loss
                    ad_loss_m.update(ad_loss, data.size(0))
                else:
                    loss = center_loss
 
                val_loss_m.update(loss)
                center_loss_m.update(center_loss, data.size(0))

                # target = torch.argmax(target,axis=1)
                # dist = torch.abs(torch.tanh(torch.sum(torch.pow(output[0]-0,2),axis=1)))
                # pred = torch.where(dist >= 0.5, torch.zeros_like(dist), torch.ones_like(dist))

                pred_score = nn.Softmax(dim=1)(output[0])
                if len(target.shape)>1:
                    target = target.max(1, keepdim=True)[1] 
                pred = pred_score.max(1, keepdim=True)[1] # get the index of the max log-probability
                
                # batch_pred_score = pred.data.cpu().numpy().tolist()
                # batch_label_score = target.data.cpu().numpy().tolist()
                # pres.extend(batch_pred_score)
                # labels.extend(batch_label_score)
                
                val_acc = accuracy_score(y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy())
                val_acc_m.update(val_acc)

        #print('\n',output[0],img_names[0])
        # pres = np.array(pres)
        # labels = np.array(labels)

        self.best_score = val_acc_m.avg
        if 'F1' in self.cfg['metrics']:
            precision, recall, f1_score = getF1(pres, labels)
            print(' \n           [VAL] loss: {:.5f}, acc: {:.3f}%, precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}\n'.format(
                val_loss_m.avg, 100. * val_acc_m.avg, precision, recall, f1_score))
            self.best_score = f1_score 

        else:
            print(' \n           [VAL] loss: {:.5f}, center_loss: {:.4f}, ad_loss: {:.4f}, acc: {:.3f}% \n'.format(
                val_loss_m.avg, center_loss_m.avg, ad_loss_m.avg, 100. * val_acc_m.avg))

        if self.cfg['warmup_epoch']:
            self.scheduler.step(epoch)
        else:
            if 'default' in self.cfg['scheduler']:
                self.scheduler.step(self.best_score)
            else:
                self.scheduler.step()


        self.checkpoint(epoch)
        if self.cfg["use_early_stop"]: self.earlyStop(epoch)


    def onTest(self):
        self.model.eval()
        
        #predict
        res_list = []
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list


    def earlyStop(self, epoch):
        ### earlystop
        if self.best_score>self.early_stop_value:
            self.early_stop_value = self.best_score
            self.early_stop_dist = 0

        self.early_stop_dist+=1
        if self.early_stop_dist>self.cfg['early_stop_patient']:
            self.best_epoch = epoch-self.cfg['early_stop_patient']+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.cfg['early_stop_patient'],self.best_epoch,self.early_stop_value))
            self.earlystop = True
        if  epoch+1==self.cfg['epochs']:
            self.best_epoch = epoch-self.early_stop_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d : %f" % (self.best_epoch,self.early_stop_value))
            self.earlystop = True

    def checkpoint(self, epoch):
        
        if self.best_score<=self.early_stop_value:
            if self.cfg['save_best_only']:
                pass
            else:
                save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'],epoch+1,self.best_score)
                self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
                self.modelSave(self.last_save_path)
        else:
            if self.cfg['save_one_only']:
                if self.last_save_path is not None and os.path.exists(self.last_save_path):
                    os.remove(self.last_save_path)
            save_name = '%s_e%d_%.5f.pth' % (self.cfg['model_name'],epoch+1,self.best_score)
            self.last_save_path = os.path.join(self.cfg['save_dir'], save_name)
            self.modelSave(self.last_save_path)


    def modelLoad(self,model_path, data_parallel = True):
        self.model.load_state_dict(torch.load(model_path), strict=True)
        
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def modelSave(self, save_name):
        torch.save(self.model.state_dict(), save_name)

    def toOnnx(self, save_name= "model.onnx"):
        dummy_input = torch.randn(1, 3, self.cfg['img_size'][0], self.cfg['img_size'][1]).to(self.device)

        torch.onnx.export(self.model, 
                        dummy_input, 
                        os.path.join(self.cfg['save_dir'],save_name), 
                        verbose=True)


