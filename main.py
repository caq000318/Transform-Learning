from __future__ import print_function,division
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()


if __name__ == '__main__':

    def imshow(inp, title=None):                                                        #可视化训练图像函数
        inp = inp.numpy().transpose((1, 2, 0))                                          #transpose：矩阵转置，012轴交错
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean                                                          #归一标准化？
        inp = np.clip(inp, 0, 1)                                                        #clip:限定数组极值，小于0的变成0，大于1的变成1
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.0001)

    #--------------通用模型训练分割线-----------------------------------------------------

    def train_model(model,criterion,optimizer,scheduler,num_epochs=25):                 #基础训练模型函数
        since=time.time()                                                               #time.time():返回时间戳

        best_model_wts=copy.deepcopy(model.state_dict())                                #deepcopy:正常复制，与copy()相对
                                                                                        #state_dict:字典对象，在建立model后自动生成
        best_acc=0.0

        for epoch in range(num_epochs):                                                 #测试25组不同的学习率与模型
            print('Epoch {}/{}'.format(epoch,num_epochs-1))                             #输出当前次数
            print('-'*10)

            for phase in ['train','val']:
                if phase=='train':
                    model.train()                                                       #model.train():限定学习模式
                else:
                    model.eval()                                                        #model.eval():限定测试模式

                running_loss=0.0
                running_corrects=0

                for inputs,labels in dataloaders[phase]:                                #迭代dataloaders中的图片
                    inputs=inputs.to(device)                                            #to(device)：将指定变量拷贝至device，之后的工作都在gpu上进行
                    labels=labels.to(device)

                    optimizer.zero_grad()                                               #optimizer.zero_grad()：梯度置零，即把loss对w的导数归零

                    with torch.set_grad_enabled(phase=='train'):                        #set_grad_enabled:会将在这个with包裹下的所有的计算出的新的变量的required_grad 置为false
                        outputs=model(inputs)                                           #构建网络
                        _,preds=torch.max(outputs,1)                                    #torch.max()：返回value和index，此处不需要value，故用_存储。
                                                                                        # 参数dim=1，表示求所在行的最大值
                        loss=criterion(outputs,labels)                                  #计算损失函数的值

                        if phase=='train':
                            loss.backward()                                             #反向传播
                            optimizer.step()                                            #参数做一次更新

                    running_loss+=loss.item()*inputs.size(0)                            #累加损失函数
                    running_corrects+=torch.sum(preds==labels.data)                     #累加正确次数

                if phase=='train':
                    scheduler.step()                                                    #对learing_rate进行调整

                epoch_loss=running_loss/dataset_sizes[phase]                            #计算损失函数平均值
                epoch_acc=running_corrects.double()/dataset_sizes[phase]                #计算平均正确次数，double():转化为double类型？？？

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase,epoch_loss,epoch_acc
                ))

                if phase=='val' and epoch_acc>best_acc:
                    best_acc=epoch_acc                                                  #保留最优的学习率
                    best_model_wts=copy.deepcopy(model.state_dict())                    #保留最优的模型

            print()

        time_elapsed=time.time()-since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)                                           #返回最优模型
        return model
    #------------------模型预测可视化分割线--------------------------------------------

    def visualize_model(model,num_images=6):                                            #输出6幅图片
        was_training=model.training
        model.eval()
        images_so_far=0
        fig=plt.figure()                                                                #创建空白图片

        with torch.no_grad():
            for i,(inputs,labels) in enumerate(dataloaders['val']):                     #enumerate：把元素和下标组合，此处用i存储下标
                inputs=inputs.to(device)
                labels=labels.to(device)

                outputs=model(inputs)
                _,preds=torch.max(outputs,1)

                for j in range(inputs.size()[0]):
                    images_so_far+=1
                    ax=plt.subplot(num_images//2,2,images_so_far)                       #subplot：绘图，三个参数表示行数，列数，把图片放在第几幅
                    ax.axis('off')                                                      #关闭坐标轴
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))         #设置标题
                    imshow(inputs.cpu().data[j])

                    if images_so_far==num_images:
                        model.train(model==was_training)
                        return

            model.train(model==was_training)

    #-------------------主程序分割线-------------------------------------------------


    data_transforms={
        'train':transforms.Compose([                                                    #compose:对多个图片进行串联操作
            transforms.RandomResizedCrop(224),                                          #randomresizedcrop：对图片进行随机裁剪，并缩放至224*224
            transforms.RandomHorizontalFlip(),                                          #randomhorizontalflip:按照概率p对图片进行水平翻转，默认概率p=0.5
            transforms.ToTensor(),                                                      #totensor：转化为tensor类型（tensor：CHW多维数组），然后每个像素除以255
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])               #normalize：对chw数据按通道进行标准化，即先减均值，再除以标准差
        ]),
        'val':transforms.Compose([
            transforms.Resize(256),                                                     #resize:重置图像分辨率，参数size，如果height>width,
                                                                                            # 则重置为(size*height/width,size)，所以建议size设定为h*w
            transforms.CenterCrop(224),                                                 #centercrop：依据给定的size从中心裁剪为（size，size）,此处剪为224*224
            transforms.ToTensor(),                                                      #归一化，标准化
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

    data_dir=r'C:\Program Files (x86)\py_work\teaching\hymenoptera_data'                #数据目录
    image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),                    #os.path.join:路径拼接函数
                                            data_transforms[x])                          #datasets.imagefolder:参数root，文件路径。参数transform，对图片预处理的函数。
                                                                                            #返回一个datasets类型
                    for x in ['train','val']}
    dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],                       #torch.utils.data.DataLoader：将datasets封装成迭代器，可以输入/输出datasets的内容
                                                                                            # 参数datasets：数据集
                                            batch_size=4,                                    #参数batch_size:每次迭代时输入/输出的行数，默认为1
                                            shuffle=True,                                    #参数shuffle:是否打乱顺序后输入/输出，默认为false
                                            num_workers=4)                                   #参数num_workers:使用多少个子进程来导入数据，默认为0.
                for x in['train','val']}
    dataset_sizes={x:len(image_datasets[x])                                             #求数据集大小
                for x in['train','val']}
    class_names=image_datasets['train'].classes                                         #百度不到，似乎是返回“bees""ants"，即返回下一级文件夹的名字

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")               #torch.device():设置为用cuda工作
    #print(device)                                                                      #检验是用gpu还是cpu工作


    inputs,classes=next(iter(dataloaders['train']))                                     #注意：多线程迭代只能在main下运行，否则会报错
    out=torchvision.utils.make_grid(inputs)                                             #make_grid：将多幅图像合并成一幅，默认一行8幅
    imshow(out,title=[class_names[x] for x in classes])

    #----------------------迁移方式1：微调convnet---------------------------------------

    model_ft=models.resnet18(pretrained=True)                                           #使用resnet18模型，pretrained=true表示采用预训练参数
    num_ftrs=model_ft.fc.in_features                                                    #获取全连接层的输入特征

    model_ft.fc=nn.Linear(num_ftrs,2)                                                   #nn.linear：设置全连接层

    model_ft=model_ft.to(device)

    criterion=nn.CrossEntropyLoss()                                                     #crossentropyloss：交叉熵损失函数

    optimizer_ft=torch.optim.SGD(model_ft.parameters(),lr=0.001,                        #SGD：实现随机梯度下降
                           momentum=0.9)                                                #参数momentum：动量因子
    exp_lr_scheduler=lr_scheduler.StepLR(optimizer_ft,step_size=7,                      #steplr：调整学习率
                                         gamma=0.1)

    model_ft=train_model(model_ft,criterion,optimizer_ft,                               #训练模型
                         exp_lr_scheduler,num_epochs=25)

    visualize_model(model_ft)                                                           #模型预测可视化
    '''
    plt.ioff()
    plt.show()
    '''
    #---------------------迁移方式2:ConvNet作为固定功能提取器---------------------------------

    model_conv=torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad=False

    num_ftrs=model_conv.fc.in_features
    model_conv.fc=nn.Linear(num_ftrs,2)

    model_conv=model_conv.to(device)

    criterion=nn.CrossEntropyLoss()

    optimizer_conv=torch.optim.SGD(model_conv.fc.parameters(),lr=0.001,
                                   momentum=0.9)

    exp_lr_scheduler=lr_scheduler.StepLR(optimizer_conv,step_size=7,
                                         gamma=0.1)

    model_conv=train_model(model_conv,criterion,optimizer_conv,
                           exp_lr_scheduler,num_epochs=25)

    visualize_model(model_conv)

    plt.ioff()
    plt.show()
