
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

###########################  loss

def labelSmooth(one_hot, label_smooth):
    return one_hot*(1-label_smooth)+label_smooth/one_hot.shape[1]


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


class CrossEntropyLossV2(nn.Module):
    def __init__(self, label_smooth=0, class_weight=None):
        super().__init__()
        self.class_weight = class_weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7
        
    def forward(self, x, y, label_smooth=0, gamma=0, sample_weights=None, sample_weight_img_names=None):

        #one_hot_label = F.one_hot(y, x.shape[1])
        one_hot_label = y
        if label_smooth:
            one_hot_label = labelSmooth(one_hot_label, label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -one_hot_label * y_softmaxlog

        if class_weight:
            loss = loss*self.class_weight

        #focal loss gamma
        if gamma:
            loss = loss*((1-y_softmax)**gamma)

        loss = torch.mean(torch.sum(loss, -1))

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth=0, class_weight=None):
        super().__init__()
        self.class_weight = class_weight 
        self.label_smooth = label_smooth
        self.epsilon = 1e-7
        
    def forward(self, x, y, sample_weights=0, sample_weight_img_names=None,gamma=None):


        if self.label_smooth:
            y = labelSmooth(y, self.label_smooth)

        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        # original CE loss
        loss = -y * y_softmaxlog

        if self.class_weight:
            loss = loss*self.class_weight

        if gamma:
            loss = loss*((1-y_softmax)**gamma)

        loss = torch.mean(torch.sum(loss, -1))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, label_smooth=0, gamma = 0., weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight # means alpha
        self.epsilon = 1e-7
        self.label_smooth = label_smooth

        
    def forward(self, x, y, sample_weights=0, sample_weight_img_names=None):

        if len(y.shape) == 1:
            #
            one_hot_label = F.one_hot(y, x.shape[1])

            if self.label_smooth:
                one_hot_label = labelSmooth(one_hot_label, self.label_smooth)

            if sample_weights>0 and sample_weights is not None:
                #print(sample_weight_img_names)
                weigths = [sample_weights  if 'yxboard' in img_name  else 1 for img_name in sample_weight_img_names] 
                weigths = torch.DoubleTensor(weigths).reshape((len(weigths),1)).to(x.device)
                #print(weigths, weigths.shape)
                #print(one_hot_label, one_hot_label.shape)
                one_hot_label = one_hot_label*weigths
                #print(one_hot_label)
                #b
        else:
            one_hot_label = y


        #y_pred = F.log_softmax(x, dim=1)
        # equal below two lines
        y_softmax = F.softmax(x, 1)
        #print(y_softmax)
        y_softmax = torch.clamp(y_softmax, self.epsilon, 1.0-self.epsilon)# avoid nan
        y_softmaxlog = torch.log(y_softmax)

        #print(y_softmaxlog)
        # original CE loss
        loss = -one_hot_label * y_softmaxlog
        #loss = 1 * torch.abs(one_hot_label-y_softmax)#my new CE..ok its L1...

        # print(one_hot_label)
        # print(y_softmax)
        # print(one_hot_label-y_softmax)
        # print(torch.abs(y-y_softmax))
        #print(loss)
        
        # gamma
        loss = loss*((torch.abs(one_hot_label-y_softmax))**self.gamma)
        # print(loss)

        # alpha
        if self.weight is not None:
            loss = self.weight*loss

        loss = torch.mean(torch.sum(loss, -1))
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

import math
def _wing_loss(landmarks, labels, w=10.0, epsilon=2.0, weights=1.):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """

    x = landmarks - labels
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(
        torch.greater(torch.tensor(w).to(x.device), absolute_x),
        w * torch.log(1.0 + absolute_x / epsilon),
        absolute_x - c
    )
    # losses = losses * cfg.DATA.weights
    loss = torch.sum(torch.mean(losses*weights, axis=[0]))

    return loss

class OneCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(OneCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(1, feat_dim))
        self.centerlossfunc = OneCenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)

        feat_masked = feat[label.bool()]
        mask_label = torch.zeros((len(feat_masked),)).to(feat.device)
        batch_size_tensor = feat_masked.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat_masked, mask_label, self.centers, batch_size_tensor)
        return loss

class OneCenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())

        loss1 =  (feature - centers_batch).pow(2).sum() / 2.0 / batch_size
        #loss2 = _wing_loss(feature,centers_batch)
        return loss1#+loss2*1.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CEandCenter(nn.Module):
    def __init__(self, class_num, feat_dim, size_average=True, label_smooth=0, class_weight=None, center_weight=1):
        super().__init__()
        self.ce = CrossEntropyLoss(label_smooth, class_weight) 
        self.center = OneCenterLoss(class_num, feat_dim, size_average=size_average)
        self.center_weight = center_weight
        
    def forward(self, x, y, feat):


        ce_loss = self.ce(x,y,gamma=2)
        #print(y.shape)
        y = torch.argmax(y,axis=1)
        center_loss = self.center(y,feat)

        loss = ce_loss+self.center_weight*center_loss
        #print(ce_loss,center_loss)
        # bb
        return loss


class NewOneCenterLoss(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.ce = CrossEntropyLoss(0, None) 

        
    def forward(self, x, y, feat, center):
        center = center.to(x.device)

        ce_loss = self.ce(x,y,gamma=2)

        y = torch.argmax(y,axis=1)
        data = torch.sum(torch.pow(feat-center,2),axis=1)

        pos = data[y==1]
        if len(pos)>0:
            loss_pos = torch.mean(pos)
        else:
            loss_pos = 0
        neg = data[y!=1]
        if len(neg)>0:
            mean_neg = torch.mean(neg)
            # k = 1
            # loss_neg = k / (k + mean_neg)
            # loss_neg = torch.exp(-mean_neg)
            # 防止出现负值
            loss_neg = abs(1 - torch.log(mean_neg + 1))
        else:
            loss_neg = 0 #防止最后一个batch为nan

        center_loss = loss_pos+ loss_neg
        #print(torch.mean(pos),k / (k + mean_neg),pos.shape,len(neg),mean_neg)
        # bb

        loss = ce_loss+center_loss*0.5
        # print(ce_loss,center_loss)
        # bb
        return loss

class OneCenter(nn.Module):
    def __init__(self, class_num, foc_weight=0.8):
        super().__init__()
        self.ce = CrossEntropyLoss(0, None)
        self.foc_weight = foc_weight
        
    def forward(self, x, y_onehot, feat):
        ce_loss = self.ce(x,y_onehot,gamma=2)

        y = torch.argmax(y_onehot,axis=1)
        data = torch.sqrt(torch.sum(torch.pow(feat-0,2),axis=1))/1.42

        gamma = 2
        data = data*((torch.abs(data-(1-y)))**gamma)

        #print(feat, data)
        pos = data[y==1]
        if len(pos)>0:
            loss_pos = torch.mean(pos)
        else:
            loss_pos = 0
        neg = data[y!=1]
        if len(neg)>0:
            mean_neg = torch.mean(neg)
            loss_neg = 1-mean_neg
            # k = 1
            # loss_neg = k / (k + mean_neg)
            # loss_neg = torch.exp(-mean_neg)
            # loss_neg = 1 - torch.log(mean_neg + 1)
        else:
            loss_neg = 0 #防止最后一个batch为nan
        #print(loss_pos,loss_neg)
        center_loss = loss_pos+ loss_neg
        #print(torch.mean(pos),k / (k + mean_neg),pos.shape,len(neg),mean_neg)
        # bb

        # 默认0.8
        loss = ce_loss+center_loss*self.foc_weight
        # y = torch.argmax(y_onehot,axis=1)
        # dist = torch.sum(torch.pow(feat-0,2),axis=1) #0-正无穷
        
        # dist = torch.tanh(dist) #0-1
        # gamma = 0.5
        # #print(dist)
        # eps = 1e-6
        # dist = torch.clamp(dist, eps, 1.0 - eps)
        # #torch.clamp 函数将 dist 强制限制在一个较小的范围内，避免了出现非法的运算。eps 是一个很小的正数，用来保证 dist 不会特别接近于 0 或 1
        # #防止nan
        # dist = dist*(torch.abs((1-dist)-y))**gamma
        # #print(dist)
        # true = dist[y==1] #true
        # if len(true)>0:
        #     loss_true = torch.mean(true)
        # else:
        #     loss_true = 0
        # fake = dist[y!=1] #fake
        # if len(fake)>0:
        #     loss_fake = torch.mean(fake)
        #     loss_fake = 1-loss_fake
        #     # k = 1
        #     # loss_neg = k / (k + mean_neg)
        # else:
        #     loss_fake = 0 #防止最后一个batch为nan

        # #print(loss_true,loss_fake,dist)
        # loss = loss_true+loss_fake

        return loss


if __name__ == '__main__':



    device = torch.device("cpu")

    #x = torch.randn(2,2)
    x = torch.tensor([[0.1,0.7,0.2]])
    y = torch.tensor([1])
    print(x)

    loss_func = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_func(x,y)
    print("loss1: ",loss)

    # loss_func = Focalloss().to(device)
    # loss = loss_func(x,y)
    # print("loss2: ",loss)
    

    weight_loss = torch.DoubleTensor([1,1,1]).to(device)
    loss_func = FocalLoss(gamma=0, weight=weight_loss).to(device)
    loss = loss_func(x,y)
    print("loss3: ",loss)
    

    # weight_loss = torch.DoubleTensor([2,1]).to(device)
    # loss_func = Focalloss(gamma=0.2, weight=weight_loss).to(device)
    # loss = loss_func(x,y)
    # print("loss4: ",loss)