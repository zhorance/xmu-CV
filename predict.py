
from my_dataset import mydataset
import torch
from d2l import torch as d2l
import utils
import time

def evaluate(model, data_loader, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

if __name__ == '__main__':
    root = '.\\data'
    batch_size = 4
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    num_classes = 4

    test_dataset = mydataset(root, name="test_02")

    #test_dataset = mydataset(root, name="test")

    num_workers = d2l.get_dataloader_workers()

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True,
                                                 num_workers=num_workers
                                                 )
    start = time.time()

    net = torch.load('./net/net4_7.pth')
    print("第 X 组测试集")
    confmat = evaluate(net, test_loader, num_classes=num_classes)

    confmat.printtable()
    end = time.time()
    t=end-start
    print('time: {:.5f}s'.format(t))