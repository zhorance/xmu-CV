
import torch
import torchvision
from torch import nn
import utils
from unet import UNet
from torch.nn import functional as F
from my_dataset import mydataset
from d2l import torch as d2l
from dice_coefficient_loss import dice_loss, build_target

def criterion(inputs, targets, num_classes):
    loss_weight = torch.as_tensor([1.0, 2.0, 2.0, 2.0])
    loss = F.cross_entropy(inputs, targets, weight=loss_weight)
    dice_target = build_target(targets, num_classes)
    loss += dice_loss(inputs, dice_target, multiclass=True)
    return loss

def evaluate(model, data_loader, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train(root,batch_size,net, trainer, num_epochs,num_classes):


    train_dataset = mydataset(root,name="train")

    val_dataset = mydataset(root, name="val")

    num_workers = d2l.get_dataloader_workers()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=num_workers
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=num_workers
                                             )

    for epoch in range(num_epochs):
        train_loss = 0.0
        net.train()
        for i, (features, labels) in enumerate(train_loader):
            trainer.zero_grad()
            pred = net(features)
            l = criterion(pred, labels, num_classes)
            l.backward()
            trainer.step()
            train_loss = train_loss +l*batch_size
            if i%10==0:
                print("EPOCH=%d  [%d/250]    loss=%f" %(epoch,i,l))


        confmat = evaluate(net, val_loader, num_classes=num_classes)

        # val_info = str(confmat)
        print("EPOCH %d" %(epoch))
        # print(val_info)
        confmat.printtable()
        print("train_loss %.5f" %(train_loss/1000))
        torch.save(net, "net7_{}.pth".format(epoch))


if __name__ == '__main__':
    num_epochs =10
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    num_classes = 4
    batch_size = 4
    root = 'C:\\Users\\HoranCe\\Desktop\\data'
    #net = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    net = torch.load('./net/net6_9.pth')
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    train(root,batch_size,net, trainer, num_epochs,num_classes)






