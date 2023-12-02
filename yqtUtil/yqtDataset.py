# 模板
# 自定义数据集
# 可以比较灵活，这里仅做一个示例，实际设计的过程要主动地分离代码和数据
# 然后数据最好是单个文件，这样相当于压缩，读写更快
# 一般最后增减一下维度匹配上网络就可以

from torch.utils.data import Dataset
import os
import numpy as np
import torch


def getFile(root_dir, ntype="train"):
    file_list = []
    filedir = os.path.join(root_dir, ntype)
    if os.path.isdir(filedir):
        for file_ in os.listdir(filedir):
            if "txt" in file_:
                testfile = os.path.join(filedir, file_)
                file_list.append(testfile)
    file_list.sort()
    return file_list


class yqtDataset(Dataset):

    def __init__(self, root_dir, ntype="train"):
        self.root_dir = root_dir
        self.type = ntype
        self.filelist = getFile(self.root_dir, self.type)
        self.size = len(self.filelist)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        my_data = np.loadtxt(self.filelist[idx])
        x = my_data[:10]
        y = my_data[10]
        xTensor = torch.tensor(x).to(torch.float32)
        yTensor = torch.tensor(np.array([y])).to(torch.float32)

        # xTensor = torch.unsqueeze(xTensor, dim=0)

        # print(xTensor.shape)
        # print(yTensor.shape)

        return xTensor, yTensor


if __name__ == '__main__':
    testPath = "/data/Data/agri"
    aa = yqtDataset(testPath, "train")
    print(aa.filelist)
    print(aa.size)
    print(aa[0])
    print(aa[0].shape)

    aa = yqtDataset(testPath, "test")
    print(aa.size)
    print(aa[0])
