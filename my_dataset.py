import os
import torch
import json
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from d2l import torch as d2l
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
        img = self.transforms(img)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':

    root = 'C:\\Users\\HoranCe\\Desktop\\data'
    dataset = mydataset(root)
    # print(dataset[0])
    # image, mask = dataset[1]  # get some sample'
    # print(image)
    # fig, ax = plt.subplots(figsize=(14,14))
    # ax.imshow(image)
    # plt.show()
    # print(image.size)
    # print(mask)
    # batch_size = 64
    # train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
    #                                          drop_last=True,
    #                                          num_workers=d2l.get_dataloader_workers())
    # for X, Y in train_iter:
    #     print(X.shape)
    #     print(Y.shape)
    #     break
    # print(len(dataset))

    image, mask = dataset[1]
    palette_path = "./palette.json"
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    mask = mask.to("cpu").numpy().astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.putpalette(pallette)
    mask.show()