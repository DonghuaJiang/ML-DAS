import numpy as np
from modelsBN import *
from datasets import *
from nb_model import NB4D_model
import matplotlib.pyplot as plt
import os, cv2, torch, random
from torch.utils.data import DataLoader
from datasets import ImageDataset as ImageD

secret_key_1 = [6.1, 8.2, 3.5, 9.3]                                                        # 用户1的密钥
secret_key_2 = [5.2, -3.8, 9.1, 10.7]                                                      # 用户2的密钥
secret_key_3 = [7.1, 3.8, 10.1, 5.5]                                                       # 未授权的密钥
secret_key_4 = [5.2-0.000001, -3.8, 9.1, 10.7]                                             # 错误的密钥
channels, img_height, img_width = 3, 128, 128
input_shape_GAB = (channels+6, img_height, img_width)                                      # 预定义加密网络的输入维度
input_shape_GBA = (channels+1, img_height, img_width)                                      # 预定义解密网络的输入维度
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'                                                # 防止服务挂掉
n_residual_blocks_AB, n_residual_blocks_BA, batch_size = 9, 11, 1

G_AB = GeneratorResNet(input_shape_GAB, n_residual_blocks_AB)                              # 实例化加密网络，正向：真实图像 -> 加密图像
G_BA = GeneratorResNet(input_shape_GBA, n_residual_blocks_BA)                              # 实例化解密网络，反向：加密图像 -> 解密图像
G_AB.load_state_dict(torch.load("./saved_models/A2B/G_AB_28.pth", map_location = 'cpu'))   # 配置训练好的模型参数，并将网络加载到CPU上
G_BA.load_state_dict(torch.load("./saved_models/B2A/G_BA_28.pth", map_location = 'cpu'))
transform = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]                        # 对数据集的预处理
dataloader = DataLoader(ImageD("./test_image/dataset/", transforms_=transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)               # 加载用于验证的数据集

seed_len = divmod(img_height*img_width, 8)
tmp_1 = NB4D_model(seed_len[0], secret_key_1, img_height, img_width)                       # 调用龙格库塔法求解混沌系统的轨迹
mask_T1 = torch.Tensor(np.expand_dims(tmp_1, 0).repeat(1, axis=0))                         # 扩展复制，生成一个batch的密钥掩膜
tmp_2 = NB4D_model(seed_len[0], secret_key_2, img_height, img_width)
mask_T2 = torch.Tensor(np.expand_dims(tmp_2, 0).repeat(1, axis=0))
tmp_3 = NB4D_model(seed_len[0], secret_key_3, img_height, img_width)
mask_F = torch.Tensor(np.expand_dims(tmp_3, 0).repeat(1, axis=0))
tmp_4 = NB4D_model(seed_len[0], secret_key_4, img_height, img_width)
mask_W = torch.Tensor(np.expand_dims(tmp_4, 0).repeat(1, axis=0))
# random_noise = torch.empty((1, channels, img_height, img_width)).uniform_(-1, 1)         # 产生均匀噪声
# gaussian_noise = (0.2**0.5)*torch.randn((1, channels, img_height, img_width))            # 产生均值为0，方差为0.1高斯噪声

with torch.no_grad():                                                                      # 设置当前计算不需要反向传播，并强制后边的内容不进行计算图的构建
    G_AB.eval()                                                                            # eval()：开启评估模式，并且BatchNorm层，Dropout层等用于优化训练而添加的网络层会被关闭。
    G_BA.eval()

    for _, batch in enumerate(dataloader):
        bat_img1 = batch["A"]                                                              # 获取验证数据集中的两幅图像
        bat_img2 = batch["B"]
        bat_img3 = batch["C"]
        real_A = torch.unsqueeze(bat_img1[0], dim=0)
        real_B = torch.unsqueeze(bat_img2[0], dim=0)
        real_C = torch.unsqueeze(bat_img3[0], dim=0)

    imgPK = torch.concatenate((real_A, real_C, mask_T1, mask_T2, mask_F), axis=1)
    cipher_fake, _ = G_AB(imgPK)                                                           # 密文图像
    # cipher_fake += gaussian_noise                                                        # 添加噪声
    fake_AK = torch.concatenate((cipher_fake, mask_T1), axis=1)
    fake_B_F1, _ = G_BA(fake_AK)                                                           # 第一层解密的图像
    fake_BK = torch.concatenate((cipher_fake, mask_T2), axis=1)
    fake_B_F2, _ = G_BA(fake_BK)                                                           # 第二层解密的图像
    fake_CK = torch.concatenate((cipher_fake, mask_F), axis=1)
    fake_B_F3, _ = G_BA(fake_CK)                                                           # 解密误导图像
    fake_DK = torch.concatenate((cipher_fake, mask_W), axis=1)
    fake_B_F4, _ = G_BA(fake_DK)                                                           # 错误的图像

    cipher_fake = np.squeeze(cipher_fake.detach().numpy())                                 # np.squeeze()：用于对numpy数据进行降维 (1,3,128,128) -> (3,128,128)
    cipher_fake = cipher_fake.transpose((1, 2, 0))*0.5+0.5                                 # 将数据映射至(0,1)之间
    fake_B_F1 = np.squeeze(fake_B_F1.detach().numpy())
    fake_B_F1 = fake_B_F1.transpose((1, 2, 0))*0.5+0.5                                     # (-1,1)*0.5+0.5 = (0,1)
    fake_B_F2 = np.squeeze(fake_B_F2.detach().numpy())
    fake_B_F2 = fake_B_F2.transpose((1, 2, 0))*0.5+0.5
    fake_B_F3 = np.squeeze(fake_B_F3.detach().numpy())
    fake_B_F3 = fake_B_F3.transpose((1, 2, 0))*0.5+0.5
    fake_B_F4 = np.squeeze(fake_B_F4.detach().numpy())
    fake_B_F4 = fake_B_F4.transpose((1, 2, 0))*0.5+0.5

    PI_tmp = np.squeeze(np.asarray(real_A)*0.5+0.5).transpose((1, 2, 0))
    PI = cv2.cvtColor(255*PI_tmp, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Plain image.png", PI)                                           # 保存明文图像
    MI_tmp = np.squeeze(np.asarray(real_C)*0.5+0.5).transpose((1, 2, 0))
    MI = cv2.cvtColor(255*MI_tmp, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Misleading image.png", MI)                                      # 保存误导图像
    DI_1 = cv2.cvtColor(255*fake_B_F1, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Decrypted image 1.png", DI_1)                                   # 保存第一层解密图像
    DI_2 = cv2.cvtColor(255*fake_B_F2, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Decrypted image 2.png", DI_2)                                   # 保存第二层解密图像
    DI_3 = cv2.cvtColor(255*fake_B_F4, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Decrypted image with wrong key.png", DI_3)                      # 保存第二层解密图像
    CI = cv2.cvtColor(255*cipher_fake, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Cipher image.png", CI)                                          # 保存密文图像
    WI = cv2.cvtColor(255*fake_B_F3, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/Decrypted misleading image.png", WI)                            # 保存解密的误导图像

    # 后处理，展示实验结果
    plt.subplot(2, 4, 1)
    plt.title('Plain image')
    plt.imshow(PI_tmp)

    plt.subplot(2, 4, 2)
    plt.title('Decrypted image 1')
    plt.imshow(fake_B_F1)

    plt.subplot(2, 4, 3)
    plt.title('Encrypted image 2')
    plt.imshow(fake_B_F2)

    plt.subplot(2, 4, 4)
    plt.title('Cipher image')
    plt.imshow(cipher_fake)

    plt.subplot(2, 4, 5)
    plt.title('Decrypted misleading image')
    plt.imshow(fake_B_F3)

    plt.subplot(2, 4, 6)
    plt.title('Misleading image')
    plt.imshow(MI_tmp)

    plt.subplot(2, 4, 7)
    plt.title('Decrypted image with wronng key')
    plt.imshow(fake_B_F4)
    plt.show()