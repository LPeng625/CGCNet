# ########修改输入的图像的大小
Image_size = (1024, 1024)

# ########数据路径
ROOT = '/home/lpeng/data/deepglobe/'

# #######batch_size大小，根据训练的情况进行修改
BATCHSIZE_PER_CARD = 2
TOTAL_EPOCH = 400
INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 6
NUM_UPDATE_LR = 3
BINARY_CLASS = 1
