import random, torch
from torch.autograd import Variable                                             # torch中Variable模板


class LambdaLR:                                                                 # 定义学习率衰减方式
    def __init__(self, n_epochs, offset, decay_start_epoch):                    # 调用父类方法进行初始化
        assert (n_epochs-decay_start_epoch) > 0, "衰减必须在训练结束前开始!"        # 断言过程
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):                                                      # 前向传播函数
        return 1.0-max(0,epoch+self.offset-self.decay_start_epoch)/(self.n_epochs-self.decay_start_epoch)