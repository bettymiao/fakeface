import torch
import torch.nn as nn

import pretrainedmodels

from fire.models.myefficientnet_pytorch import EfficientNet
from fire.models.convnext import convnext_tiny,convnext_small,convnext_base,convnext_large

from timm.models import create_model
from timm.models import load_checkpoint
from timm.models.layers import SelectAdaptivePool2d, ClassifierHead

import torchvision

class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        

        self.pretrainedModel()
        
        self.changeModelStructure()
        


    def pretrainedModel(self):


        ### Create model

        if "efficientnet" in self.cfg['model_name']:
            #model = EfficientNet.from_name(model_name)
            self.pretrain_model = EfficientNet.from_name(self.cfg['model_name'].replace('adv-',''))
            if self.cfg['pretrained']:
                self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=True) 

        
        elif 'resnet' in self.cfg['model_name']:
            #model_name = 'resnext50' # se_resnext50_32x4d xception
            self.pretrain_model = pretrainedmodels.__dict__[self.cfg['model_name']](num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[self.cfg['model_name']])

            if self.cfg['pretrained']:
                self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained']),strict=False)
                


        elif "convnext" in self.cfg['model_name']:
            if "base" in self.cfg['model_name']:
                self.pretrain_model = convnext_base()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
                # print(self.pretrain_model)
                # b
            elif "tiny" in self.cfg['model_name']:
                self.pretrain_model = convnext_tiny()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
                # print(self.pretrain_model)
                # b
            elif "small" in self.cfg['model_name']:
                self.pretrain_model = convnext_small()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
            elif "large" in self.cfg['model_name']:
                self.pretrain_model = convnext_large()
                if self.cfg['pretrained']:
                    self.pretrain_model.load_state_dict(torch.load(self.cfg['pretrained'])['model'],strict=False) 
        
        elif "swin" in self.cfg["model_name"]:
            self.pretrain_model = create_model(
                self.cfg["model_name"],
                pretrained=True,
                num_classes=self.cfg['class_number'],
                drop_rate=self.cfg["dropout"])
            print(self.pretrain_model.default_cfg['classifier'])
            
        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass


        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def changeModelStructure(self):
        ### Change model
        if "efficientnet" in self.cfg['model_name']:
            #self.pretrain_model._dropout = nn.Dropout(0.5)
            fc_features = self.pretrain_model._fc.in_features 
            self.pretrain_model._fc = nn.Linear(fc_features,  self.cfg['class_number'])


        elif "convnext" in self.cfg['model_name']:

            self.backbone =  self.pretrain_model
            #print(self.backbone)
            num_features = 1024
            if "large" in self.cfg['model_name']:
                num_features = 1536
            elif "tiny" in self.cfg['model_name']:
                num_features = 768
            elif "small" in self.cfg['model_name']:
                num_features = 768

            self.head1 = nn.Sequential(
                         # nn.Dropout(0.5),
                         nn.Linear(num_features,self.cfg['class_number']))


        elif 'resnet' in self.cfg['model_name']:
            #self.avgpool = nn.AdaptiveAvgPool2d(1)
            #print(self.pretrain_model)
            fc_features = self.pretrain_model.last_linear.in_features

            self.pretrain_model = nn.Sequential(*list(self.pretrain_model.children())[:-2])

            self.avgpool = nn.AdaptiveAvgPool2d(1)

            # self.head1 = nn.Linear(fc_features, self.cfg['class_number']) 
            
            # feat_dim = 2
            # self.head1 = nn.Linear(fc_features, feat_dim) 
            self.se = SE_Block(fc_features)

            feat_dim = 3
            self.head1 = nn.Linear(fc_features, feat_dim) 
            self.bn = nn.BatchNorm1d(feat_dim)
            self.tanh = nn.Tanh()
            self.preluip1 = nn.PReLU()
            self.head2 = nn.Linear(feat_dim, self.cfg['class_number']) 

            #self.centers = torch.randn(1, 2)
            #nn.Parameter(torch.randn(1, feat_dim))
            
        elif "swin" in self.cfg["model_name"]:
            fc_features = self.pretrain_model.last_linear.in_features
            
            feat_dim = 3
            self.head1 = nn.Linear(fc_features, feat_dim) 
            self.bn = nn.BatchNorm1d(feat_dim)
            self.tanh = nn.Tanh()
            self.preluip1 = nn.PReLU()
            self.head2 = nn.Linear(feat_dim, self.cfg['class_number']) 
            
        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def forward(self, img):        

        if "efficientnet" in self.cfg['model_name']:
            out = self.backbone(img)
            out = out.view(out.size(0), -1)
            out1 = self.head1(out)

            out = [out1]

        elif "convnext" in self.cfg['model_name']:

            out = self.backbone(img)
            out = out.view(out.size(0), -1)
            print(out.shape)
            out1 = self.head1(out)

            out = [out1]

        elif 'resnet' in self.cfg['model_name']:
            out = self.pretrain_model(img)
            #out = self.se(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)

            
            
            out1 = self.head1(out)
            out1 = self.bn(out1)
            out1 = self.tanh(out1)
            #print(torch.max(out),torch.min(out),torch.max(out1),torch.min(out1))
            # out = [ out1]

            #out2 = self.preluip1(out1)
            out2 = self.head2(out1)
            out = [out2, out1]


        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass

        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])

        return out


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
