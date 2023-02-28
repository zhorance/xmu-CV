from my_dataset import mydataset
import torch
from d2l import torch as d2l

import os
import torch
import json
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class mydataset(data.Dataset):
    def __init__(self, root,  name: str = "train"):
        super(mydataset, self).__init__()

        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, name, 'images')
        mask_dir = os.path.join(root, name, 'annotations')
        self.image_ids = os.listdir(image_dir)
        self.mask_ids = os.listdir(mask_dir)


        self.images = [os.path.join(image_dir, image_id) for image_id in self.image_ids]
        self.masks = [os.path.join(mask_dir, mask_id) for mask_id in self.mask_ids]
        assert (len(self.images) == len(self.masks))
        self.transforms =  transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        img1 = self.transforms(img)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return img1, target,img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    num_epochs = 10
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    num_classes = 4
    batch_size = 4
    num_workers = d2l.get_dataloader_workers()
    root = '.\\data'
    net = torch.load('./net/net4_7.pth')
    dataset = mydataset(root,'test_02')
    l= len(dataset)
    for i in range(l):
        image, mask, img = dataset[i]
        image = torch.unsqueeze(image, 0)
        output = net(image)
        output = output.argmax(1)
        output = torch.squeeze(output)
        print(output.size())

        palette_path = "./palette.json"
        with open(palette_path, "rb") as f:
            pallette_dict = json.load(f)
            pallette = []
            for v in pallette_dict.values():
                pallette += v

        mask = mask.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.putpalette(pallette)
        #print(mask)
        # print(mask.size)
        # mask.show()

        output = output.to("cpu").numpy().astype(np.uint8)
        output = Image.fromarray(output)
        output.putpalette(pallette)
        # print(output)
        # output.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.title.set_text('Image')
        ax2.title.set_text('Ground truth')
        ax3.title.set_text('Predict')
        ax1.imshow(img)
        ax2.imshow(mask)
        ax3.imshow(output)
        #plt.show()
        plt.savefig('./pic/test_{}.jpg'.format(i))










    # test_dataset = mydataset(root, name="val")
    # num_workers = d2l.get_dataloader_workers()
    #
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=1,
    #                                           shuffle=True,
    #                                           drop_last=True,
    #                                           num_workers=num_workers
    #                                           )
    # for image, target in test_loader:
    #     print(image.size())
    #     output = net(image)
    #     print(output.size())

