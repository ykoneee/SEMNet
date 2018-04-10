# SEMNet (这名字我觉得还行)
SENet+MobileNetV2  in pytorch

将SENet的SE block加入到了MobileNetV2中带有shortcut的bottleneck block中，以期望这样瞎搞可以有效果。(可能吗?做梦吧)  
其中SENet的reduction参数掐指一算设置为8而不是paper中的16。
