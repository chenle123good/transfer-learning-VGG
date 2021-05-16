import  torch
from    torch import optim, nn
import  visdom
import  torchvision
from    torch.utils.data import DataLoader

from    pokemon import Pokemon
# from    resnet import ResNet18
from    torchvision.models import vgg19

from    utils import Flatten
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import numpy

# 定义一些超参数
batchsz =64
# batchsz = 8
lr = 1e-3
epochs = 10

# 定义是否使用GPU
device = torch.device('cuda')
torch.manual_seed(1234)

# 加载数据集并预处理
train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
# print(len(train_db),len(val_db))
# 将数据放置迭代器
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)


# viz = visdom.Visdom()

def evalute(model, epoch,loader,string):
    model.eval()

    correct = 0
    total_size=0
    correct_list=[]

    for step,(x,y) in enumerate(loader,1):
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            total_size+=y.size(0)
            pred = logits.argmax(dim=1)

        correct += torch.eq(pred, y).sum().float().item()
        correct_list.append(correct/total_size)
        print('[epoch:%d, iter:%d] %s_Acc: %.3f%% '
              % (
                  epoch + 1, (step *batchsz),string,correct/total_size))
    return correct/total_size,correct_list,step
# class My_VGG19(nn.Module):
#     def __init__(self,num_class):
#         super(My_VGG19, self).__init__()
#         model=vgg19(pretrained=True)
#         self.classifier = nn.Sequential()
#         self.features=model
#         self.classifier=nn.Sequential(
#             # 512个feature，每个feature 7*7
#             nn.Linear(512*7*7, 1024),
#             nn.ReLU(True),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 256),
#             nn.ReLU(True),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_class))
#     def forward(self, x):
#         x = self.features(x)
#         print(x)
#         # x.size()[0]: batch size
#         x = x.view(x.size()[0], -1)
#         # print(x.shape)
#         x = self.classifier(x)
#
#         return x
#
#

def main():
    # # 调用模型
    # model = models.resnet50(pretrained=True)
    # # 提取fc层中固定的参数
    # fc_features = model.fc.in_features
    # # 修改类别为9
    # model.fc = nn.Linear(fc_features, 9)
    #
    # model = ResNet18(5).to(device)

    # 采用预训练模型权重进行训练
    trained_model = vgg19(pretrained=True).to(device)

    # print(trained_model)
    # 新构造模块的参数默认requires_grad=True,冻结模块参数用False
    i=0
    for param in trained_model.parameters():
        if i<28:
            param.requires_grad = False
        i=i+1
    print(i)
    # 取全连接层前面的所有层,冻结参数，backward（）不用梯度下降
    for param in trained_model.parameters():
        param.requires_grad = False
    # 构建自己的分类器，并搭建网络
    model = nn.Sequential(*list(trained_model.children())[:-1], #[b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(512*7*7, 5)
                          ).to(device)
    # x = torch.randn(2, 3, 224, 224)
    # print(model(x).shape)
    # model=My_VGG19(5).to(device)
    # 定义优化器和损失函数——第一种方法
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    # 定义优化器和损失函数——第二种方法
    # 选择优化方法
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # criteon = nn.CrossEntropyLoss()
    # # 学习率调整策略
    # # 每7个epoch调整一次
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    epoch_line = 0
    global_step = 0
    val_acc_line=[]
    val_acc_list=[]
    # val_acc=0
    # epoch_val=0

    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    train_loss_list = []
    train_correct_list = []
    # 开始训练
    for epoch in range(epochs):
        train_loss_all=0
        train_loss=0
        train_correct_number=0
        total_size=0
        # exp_lr_scheduler.step()
        for step, (x,y) in enumerate(train_loader,1):

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            # length=len(train_loader)

            logits = model(x)
            total_size+=y.size(0)
            train_loss = criteon(logits, y)
            train_loss_all+=train_loss
            train_loss_list.append(train_loss_all.item()/step)
            pred = logits.argmax(dim=1)
            # item()将相等的个数累加转换成numpy类型
            train_correct_number += torch.eq(pred, y).sum().float().item()
            train_correct_list.append(train_correct_number/total_size)

            # 开始梯度下降
            optimizer.zero_grad()
            # loss求导，反向
            train_loss.backward()
            # 执行梯度下降
            optimizer.step()

            # viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
            epoch_line += 1

            print('[epoch:%d, iter:%d] Loss: %.03f | train_Acc: %.3f%% '
                  % (epoch + 1, (step* batchsz), train_loss_all / step, train_correct_number / total_size))

        # 每训练完一个epoch验证一下准确率
        print("Waiting validate!")
        if epoch % 1 == 0:
            val_acc,val_acc_line,val_step = evalute(model,epoch, val_loader,'val')
            val_acc_list=val_acc_list+val_acc_line
            # val_acc_line.append(acc)
            # epoch_val += 1
            # if val_acc> best_acc:
            #     best_epoch = epoch
            #     best_acc = val_acc
            #
            #     # 保存模型的最好状态
            #     torch.save(model.state_dict(), 'best.mdl')
                # viz.line([val_acc], [global_step], win='val_acc', update='append')

    # 打印损失函数，准确率图形
    epoch_line = numpy.linspace(1, step*epochs, step*epochs)
    plt.plot(epoch_line, train_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['train_loss'])
    plt.show()
    plt.plot(epoch_line, train_correct_list)
    plt.xlabel("Epochs")
    plt.ylabel('acc')
    plt.legend(['train_acc'])
    plt.show()
    # print(val_acc_list)
    epoch_val = numpy.linspace(1, val_step*epochs, val_step*epochs)
    plt.plot(epoch_val, val_acc_list)
    plt.xlabel("Epochs")
    plt.ylabel('acc')
    plt.legend(['val_acc'])
    plt.show()

    # print('train_loss_last', train_loss_list[-1])
    # print('val_acc_last',val_acc_line[-1])
    # print('best acc:', best_acc, 'best epoch:', best_epoch)
    # model.load_state_dict(torch.load('best.mdl'))
    # print('loaded from ckpt!')

    # 最后测试一下模型准确率
    test_acc,test_acc_line,test_step = evalute(model,epoch,test_loader,'test')
    # test_val = numpy.linspace(1, test_step, test_step)
    # plt.plot(test_val, test_acc_line)
    # plt.xlabel("Epochs")
    # plt.ylabel('acc')
    # plt.legend(['test_acc'])
    # plt.show()





if __name__ == '__main__':
    main()
