import numpy as np

# 导入的numpy包在实现Adagrad算法时可能会被使用到
# 输入数据和标签
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591]

# 初始值和超参数设置
b = -150
w = 0
lr = 1
iteration = 10000
lr_b = 0
lr_w = 0

# w_list和b_list用于记录每一轮迭代的w和b值，用于绘图
w_list = [float] * iteration
b_list = [float] * iteration
np.random.seed(1)
for i in range(iteration):
    b_grad = 0
    w_grad = 0
    # 填空部分，实现Adagrad算法
    for x, y in zip(x_data, y_data):
        w_grad += 2 * x * (w * x + b - y)
        b_grad += 2 * (b + w * x - y)
    lr_b += b_grad * b_grad
    lr_w += w_grad * w_grad
    w = w - w_grad * lr / np.sqrt(lr_w)
    b = b - b_grad * lr / np.sqrt(lr_b)
    w_list[i] = w
    b_list[i] = b

# 绘图部分
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

fig = plt.figure(dpi=700)
plt.xlim(-200, -80)
plt.ylim(-4, 4)

# 设置背景
xmin, xmax = xlim = -200, -80
ymin, ymax = ylim = -4, 4
ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
                     autoscale_on=False)
X = [[4, 4], [4, 4], [4, 4], [1, 1]]
ax.imshow(X, interpolation='bicubic', cmap=cm.Spectral,
          extent=(xmin, xmax, ymin, ymax), alpha=1)
ax.set_aspect('auto')

# 绘制每一个数据点
plt.scatter(b_list, w_list, s=2, c='black', label=(lr, iteration))
plt.legend()
plt.show()
