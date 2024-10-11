import visdom
import numpy as np
import time


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        # self.plot('loss', 1.00)

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        """
        return getattr(self.vis, name)


import matplotlib.pyplot as plt

# 读取数据
epoch_loss_data = {}  # 用于存储epoch和对应的loss
epoch = None
loss = None
with open('logs/BiReNet34_baseline.log', 'r') as f:
    for line in f:
        if 'time: ' in line:
            parts = line.strip().split("time: ")[0].strip().split("epoch: ")
            epoch = int(parts[1])
        elif 'tensor(' in line:
            parts = line.strip().split(", device")[0].strip().split("tensor(")
            loss = float(parts[1])  # 假设loss是第四个元素
        if epoch is not None and loss is not None:
            epoch_loss_data[epoch] = loss
            epoch = None
            loss = None

# 将数据转换为列表形式以便于绘图
epochs = list(epoch_loss_data.keys())
losses = list(epoch_loss_data.values())

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o')
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
