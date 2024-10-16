import torch
import torch.nn as nn

import modify

def preactblock(in_channels, channels, stride=1):
    """
    Implements PreActBlock from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py 
    Note that self.expansion=1 in that code, so it is redundant.
    """
    if stride == 1 and in_channels == channels:
        shortcut = nn.Identity()
    else:
        shortcut = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)

    return modify.Sequential([
        ('bn1', modify.BatchNorm2d(in_channels)),
        nn.ReLU(),
        modify.Copy(2),
        ('parallel', modify.Parallel([
            ('shortcut', shortcut),
            modify.Sequential([
                ('conv1', nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)),
                ('bn2', modify.BatchNorm2d(channels)),
                nn.ReLU(),
                ('conv2', nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)),
                modify.PrintShape(),
            ]),
        ])),
        modify.PrintShape("After residual"),
        modify.Add(),
    ])
 
block = preactblock(3, 2)
result = block(torch.randn(1, 3, 10, 10))
