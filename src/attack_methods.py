import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
#from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math
from PIL import Image
import pickle

import pudb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs, None


class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, vit, featureExtractor, config):
        super(Attack_PGD, self).__init__()

        self.vit = vit
        self.featureExtractor = featureExtractor
        #for p in self.parameters():
        #    p.requires_grad = False
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.top_k = config['top_k']
        self.patch_num = config['patch_num']
        self.image_size = config['image_size']
        self.patch_size = int(self.image_size / self.patch_num) #16
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.attack_w = nn.Parameter(torch.FloatTensor(torch.rand(3, self.image_size, self.image_size)))   #[3, 224, 224]

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):
        aux_vit = pickle.loads(pickle.dumps(self.vit))
        aux_featureExtractor = pickle.loads(pickle.dumps(self.featureExtractor))
        for name, value in aux_vit.named_parameters():
            if "patch_embeddings" in name:
                value.requires_grad = True
        aux_vit.eval()
        aux_vit.zero_grad()
        outputs = aux_vit(inputs)
        targets_prob = F.softmax(outputs.logits.float(), dim=1) #[batch classes_size]
        #print(outputs.logits)
        outputs.patch_embeddings.retain_grad()
        loss = self.loss_func(targets_prob, targets).mean()
        loss.backward(retain_graph=True)
        #print(targets_prob.max(1), "    ", targets)
        if not attack:
            return outputs, None
        
        patch_embeddings = outputs.patch_embeddings  #[n, 196, 768]
        #pu.db
        patch_grad = patch_embeddings.grad.data.detach() #[n, 196, 768]
        sum_patch = torch.sum(torch.sum(patch_grad, dim = 0), dim = 1) #[196]
        vlaues, top_k_patch = sum_patch.topk(self.top_k)

        for name, value in self.named_parameters():
            if "attack_w" not in name:
                value.requires_grad = False

        x = inputs.detach() #[batch, 3, 224, 224]

        self.attack_w.requires_grad_()

        for i in top_k_patch:
            ind_x = int(i / self.patch_num)
            ind_y = int(i % self.patch_num)
            xx_start = ind_x * self.patch_size
            yy_start = ind_y * self.patch_size
            for xx in range(xx_start, xx_start + self.patch_size):
                for yy in range(yy_start, yy_start + self.patch_size):
                    x[:, :, xx, yy] = x[:, :, xx, yy] + self.attack_w[:, xx, yy]
        if x.grad is not None:
            x.grad.zero_()
        if x.grad is not None:
            x.grad.data.fill_(0)
        for name, value in self.named_parameters():
            if "attack_w" in name:
                value.requires_grad = True
        aux_vit.eval()
        #pu.db
        outputs = aux_vit(x)
        logits = outputs.logits
        #print(logits)
        targets_prob = F.softmax(logits.float(), dim=1)
        
        l2loss = torch.mean(torch.mul(self.attack_w, self.attack_w))
        #print(self.attack_w)
        u = 0
        loss = u * l2loss - (1 - u) * self.loss_func(targets_prob, targets).mean()
        #print(u * l2loss, loss)
        #loss.backward(retain_graph=True)
        #print(self.attack_w.grad)
        return targets_prob, loss

