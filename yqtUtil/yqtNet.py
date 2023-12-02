# 模板
# 定义网络结构

import torch
import torch.nn as nn


class yqtNet(nn.Module):

    def __init__(self):
        super(yqtNet, self).__init__()
        self.l1 = nn.Linear(10, 1)
        self.l2 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


if __name__ == '__main__':
    net = yqtNet()
    input_ = torch.randn(4, 10)

    out = net(input_)
    print(out)
    print(out.shape)
