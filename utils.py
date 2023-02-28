import torch

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def printtable(self):
        acc_global, acc, iu = self.compute()
        iou = ['{:.5f}'.format(i) for i in (iu * 100).tolist()]
        cls_id = [0, 1, 2, 3]
        cls_name = ["background", "bubble", "scratch", "tin_ash"]
        d = []

        for i in range(4):
            l = []
            l.append(cls_id[i])
            l.append(cls_name[i])
            l.append(iou[i])
            d.append(l)

        print("{:<8} {:<15} {:<10}".format('cls_id', 'cls_name', 'IoU(%)'))
        for v in d:
            id, name, IOU = v
            print("{:<8} {:<15} {:<10}".format(id, name, IOU))
        print('mean IoU: {:.5f}'.format(iu.mean().item() * 100))


# iou = [96.12344,23.23144,45.23134,11.34545]
#
# cls_id = [0,1,2,3]
# cls_name = ["background","backbubble","scratch","tin_ash"]
# d = []
#
# print()
# print ("{:<8} {:<15} {:<10}".format('cls_id','cls_name','IoU(%)'))
#
# for i in range(4):
#     l = []
#     l.append(cls_id[i])
#     l.append(cls_name[i])
#     l.append(iou[i])
#     d.append(l)
#
# print(d)
# for v in d:
#     id, name, IOU = v
#     print ("{:<8} {:<15} {:<10}".format( id, name, IOU))