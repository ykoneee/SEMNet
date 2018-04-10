# SEMNet (这名字我觉得还行)

### SENet+MobileNetV2  in pytorch

将SENet的SE block加入到了MobileNetV2中带有shortcut的bottleneck block中  
以期望这样瞎搞可以有效果。(可能吗?做梦吧)  

其中SENet的reduction参数，我掐指一算设置为8而不是paper中的16。

#### 环境需求：
    pytorch 0.3.1
    tqdm

#### 快速开始：
    在cifar-10上进行训练，使用pytorch官方cifar10数据集
    python pytorch_baseline.py
    ![](1.png 'result')
    
#### 然后呢？
    自己看代码吧，没有教学。
