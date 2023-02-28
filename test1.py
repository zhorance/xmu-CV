cls_id = [1, 2, 3]
cls_name = [ "bubble", "scratch", "tin_ash"]
iou = [53.38188,49.61864,40.36392]
d = []
iu=(53.38188+49.61864+40.36392)/3
for i in range(3):
    l = []
    l.append(cls_id[i])
    l.append(cls_name[i])
    l.append(iou[i])
    d.append(l)


print("{:<8} {:<15} {:<10}".format('cls_id', 'cls_name', 'IoU(%)'))
for v in d:
    id, name, IOU = v
    print("{:<8} {:<15} {:<10}".format(id, name, IOU))
print('mean IoU: {:.5f}'.format(iu))