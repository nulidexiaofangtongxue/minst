import torch.nn as nn  # pytorch中最重要的模块，封装了神经网络相关的函数
import torch
import torch.nn.functional as F  # 提供了一些常用的函数，如softmax
import torch.optim as optim  # 优化模块，封装了求解模型的一些优化器，如Adam SGD
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口


# 预设网络超参数 （所谓超参数就是可以人为设定的参数

BATCH_SIZE = 64  # 由于使用批量训练的方法，需要定义每批的训练的样本数目

EPOCHS = 3  # 总共训练迭代的次数

# 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001  # 设定初始的学习率
# 加载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))  # 数据规范化到正态分布
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)  # 指明批量大小，打乱，这是处于后续训练的需要。

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)


# 设计模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 提取特征层
        self.features = nn.Sequential(
            # 卷积层
            # 输入图像通道为 1，因为我们使用的是黑白图，单通道的
            # 输出通道为32（代表使用32个卷积核），一个卷积核产生一个单通道的特征图
            # 卷积核kernel_size的尺寸为 3 * 3，stride 代表每次卷积核的移动像素个数为1
            # padding 填充，为1代表在图像长宽都多了两个像素
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # ((28-3+2*1)/1)+1=28  28*28*1  》 28*28*32

            # 批量归一化，跟上一层的out_channels大小相等，以下的通道规律也是必须要对应好的
            nn.BatchNorm2d(num_features=32),  # 28*28*32  》 28*28*32

            # 激活函数，inplace=true代表直接进行运算
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # ((28-3+2*1)/1)+1=28     28*28*32 》 28*28*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 最大池化层
            # kernel_size 为2 * 2的滑动窗口
            # stride为2，表示每次滑动距离为2个像素
            # 经过这一步，图像的大小变为1/4，即 28 * 28 -》 7 * 7
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 分类层
        self.classifier = nn.Sequential(
            # Dropout层
            # p = 0.5 代表该层的每个权重有0.5的可能性为0
            nn.Dropout(p=0.5),
            # 这里是通道数64 * 图像大小7 * 7，然后输入到512个神经元中
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    # 前向传递函数
    def forward(self, x):
        # 经过特征提取层
        x = self.features(x)
        # 输出结果必须展平成一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 初始化模型 将网络操作移动到GPU或者CPU

ConvModel = ConvNet().to(DEVICE)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(DEVICE)
# 定义模型优化器：输入模型参数，定义初始学习率
optimizer = torch.optim.Adam(ConvModel.parameters(), lr=learning_rate)
# 定义学习率调度器：输入包装的模型，定义学习率衰减周期step_size，gamma为衰减的乘法因子
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
# 在官网上的解释。如果初始学习率lr = 0.05，衰减周期step_size为30，衰减乘法因子gamma=0.01
# Assuming optimizer uses lr = 0.05 for all groups
# >>> # lr = 0.05     if epoch < 30
# >>> # lr = 0.005    if 30 <= epoch < 60
# >>> # lr = 0.0005   if 60 <= epoch < 90
def train(num_epochs, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    _model.train()  # 设置模型为训练模式
    _lr_scheduler.step() # 设置学习率调度器开始准备更新
    for epoch in range(num_epochs):
        # 从迭代器抽取图片和标签
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            # 此时样本是一批图片，在CNN的输入中，我们需要将其变为四维，
            # reshape第一个-1 代表自动计算批量图片的数目n
            # 最后reshape得到的结果就是n张图片，每一张图片都是单通道的28 * 28，得到四维张量
            output = _model(samples.reshape(-1, 1, 28, 28))
            # 计算损失函数值
            loss = criterion(output, labels)
            # 优化器内部参数梯度必须变为0
            optimizer.zero_grad()
            # 损失值后向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            if (i + 1) % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch + 1, num_epochs, i + 1, loss.item()))


def test(_test_loader, _model, _device):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0

    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = ConvModel(data.reshape(-1, 1, 28, 28))
            loss += criterion(output, target).item()  # 添加损失值
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来。

    loss /= len(_test_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))
for epoch in range(1, EPOCHS + 1):
    train(epoch, ConvModel, DEVICE, train_loader, optimizer, exp_lr_scheduler)
    test(test_loader,ConvModel, DEVICE)
    test(train_loader,ConvModel, DEVICE)