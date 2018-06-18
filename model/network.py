import torch
import torch.nn as nn
import numpy as np
from .losses.flow_loss import flow_loss
from .net.tvnet import TVNet
from torch.autograd import Variable

class model():
    def __init__(self, args):

        self.args = args

        ## Define neural networks...
        self.flow_net = TVNet(args).cuda()
        
        ## ...with their losses...
        self.flow_loss = flow_loss(args).cuda()

        ## ... and optimizers
        self.flow_optmizer = torch.optim.SGD(self.flow_net.parameters(), 
                                             lr=args.learning_rate,
                                             momentum=0.5)
    
    def forward(self, x1, x2, need_result=False):
        x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        u1, u2, rho = self.flow_net(x1, x2)
        self.loss = self.flow_loss(u1, u2, x1, x2)
        if need_result:
            return u1, u2
    
    def optimize(self):
        self.flow_optmizer.zero_grad()
        self.loss.backward()
        self.flow_optmizer.step()

    def update_optimizer(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
