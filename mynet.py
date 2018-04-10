import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
from torch.nn import init
import torch
from se_module import SELayer
class Bottleneck(nn.Module):
    def __init__(self,in_c, out_c,t,stride):
        super(Bottleneck,self).__init__()
        self.stride=stride
        self.in_c=in_c
        self.out_c=out_c
        self.block=nn.Sequential(
        nn.Conv2d(in_c, in_c * t,kernel_size=1, bias=False),
        nn.BatchNorm2d(in_c * t),
        #nn.Dropout2d(0.2),
        nn.ReLU6(True),
        nn.Conv2d(in_c*t,in_c*t,kernel_size=3, stride=stride,padding=1,
                      groups=in_c * t, bias=False),
        nn.BatchNorm2d(in_c * t),
        #nn.Dropout2d(0.2),
        nn.ReLU6(True),
        nn.Conv2d(in_c * t, out_c, kernel_size=1,bias=False),
        nn.BatchNorm2d(out_c),
        )
        self.se=SELayer(out_c, reduction=8)
        #self.sampler=nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False),nn.BatchNorm2d(out_c))
    def forward(self,x):
        res=x
        out=self.block(x)
        if self.stride==1:
            if self.in_c==self.out_c:
                return self.se(out)+res
            #else:
                #return out+self.sampler(res)
        return out
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10,in_c=3,scale=1.0):
        cfg = [ 
                {'t':None, 'c': 32, 'n': 1, 's': 1,'name':'conv2d'},#s 2->1
                {'t': 1, 'c': 16, 'n': 1, 's': 1,'name':'bottleneck1'},
                {'t': 6, 'c': 24, 'n': 2, 's': 1,'name':'bottleneck2'},#s 2->1
                {'t': 6, 'c': 32, 'n': 3, 's': 2,'name':'bottleneck3'},
                {'t': 6, 'c': 64, 'n': 4, 's': 2,'name':'bottleneck4'},
                {'t': 6, 'c': 96, 'n': 3, 's': 1,'name':'bottleneck5'},
                {'t': 6, 'c': 160, 'n': 3, 's': 2,'name':'bottleneck6'},
                {'t': 6, 'c': 320, 'n': 1, 's': 1,'name':'bottleneck7'},
                #{'t': None, 'c': 1280, 'n': 1, 's': 1,'name':'conv2d 1x1'},
                #{'t': None, 'c': None, 'n': 1, 's': 1,'name':'avgpool 7x7'},
                #{'t': None, 'c': None, 'n': 1, 's': 1,'name':'conv2d 1x1'},
        ]        
        super(MobileNetV2, self).__init__()
        self.conv1=nn.Conv2d(in_c,int(cfg[0]['c']*scale),kernel_size=3,bias=False,stride=cfg[0]['s'],padding=1)
        self.bn1=nn.BatchNorm2d(int(cfg[0]['c']*scale))
        self.bottlenecklist=[]
        for i in range(1,len(cfg)):
            self.bottlenecklist.extend(
                self._make_layers(int(cfg[i-1]['c']*scale),
                                  int(cfg[i]['c']*scale),
                                  cfg[i]['t'],cfg[i]['n'],cfg[i]['s']))
        self.B_layers=nn.Sequential(*self.bottlenecklist)
        
        self.conv2=nn.Conv2d(int(scale*cfg[-1]['c']),int(1280*scale),kernel_size=1,bias=False)
        self.bn2=nn.BatchNorm2d(int(1280*scale))      
        self.conv3=nn.Conv2d(int(1280*scale), 10, 1)
        
        self.initialize()
    def _make_layers(self,in_c,out_c,t,n,stride):
        layers=[]
        layers.append(Bottleneck(in_c, out_c, t, stride))
        for i in range(0,n-1):
            layers.append(Bottleneck(out_c,out_c,t,1))
        return layers
    def forward(self, x):
        out=self.conv1(x)
        out=self.bn1(out)
        #out=F.dropout2d(out,0.2)
        out=F.relu6(out)
        out=self.B_layers(out)
        out=self.conv2(out)
        out=self.bn2(out)
        #out=F.dropout2d(out,0.2)
        out=F.relu6(out)    
        
        #out=F.dropout2d(out,0,2)
        #out=F.avg_pool2d(out,4)# 7->4
        out=F.adaptive_avg_pool2d(out, 1)
        #out=F.dropout2d(out,0.2)
        
        out=self.conv3(out)
        
        out=out.view(out.size(0),-1)
        return out
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal(m.weight)
                #init.kaiming_normal(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
if __name__=='__main__':
    import sys
    net=MobileNetV2(in_c=3)
    inputs=Variable(torch.zeros((1,3,32,32)))
    #print(net)
    print(net(inputs).shape)