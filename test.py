iou = [96.12344,23.23144,45.23134,11.34545]

cls_id = [0,1,2,3]
cls_name = ["background","backbubble","scratch","tin_ash"]
d = []

print()
print ("{:<8} {:<15} {:<10}".format('cls_id','cls_name','IoU(%)'))

for i in range(4):
    l = []
    l.append(cls_id[i])
    l.append(cls_name[i])
    l.append(iou[i])
    d.append(l)

print(d)
for v in d:
    id, name, IOU = v
    print ("{:<8} {:<15} {:<10}".format( id, name, IOU))

iu = [12,345,3674,8467588,87696,124]

iou = ['{:.5f}'.format(i) for i in iu ]

print(iou)