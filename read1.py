

import os
import torch
import json
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt




if __name__ == '__main__':
    dir_img = '.\\data\\val\\images\\1643.jpg'
    dir_annotation = '.\\data\\val\\annotations\\1643.png'

    transforms = transforms.Compose([
                     transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img = Image.open(dir_img).convert('RGB')
    image = transforms(img)
    target = Image.open(dir_annotation)
    mask = torch.as_tensor(np.array(target), dtype=torch.int64)

    net = torch.load('./net/net5_6.pth')

    image = torch.unsqueeze(image, 0)
    output = net(image)
    output = output.argmax(1)
    output = torch.squeeze(output)
    #print(output.size())

    palette_path = "./palette.json"
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    mask = mask.to("cpu").numpy().astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.putpalette(pallette)
    print(mask)
    #print(mask.size)
    #mask.show()

    output = output.to("cpu").numpy().astype(np.uint8)
    output = Image.fromarray(output)
    output.putpalette(pallette)
    #print(output)
    #output.show()

    # plt.subplot(1, 3, 1)
    # plt.imshow(img)
    # plt.subplot(132)
    # plt.imshow(mask)
    # plt.subplot(133)
    # plt.imshow(output)
    # plt.show()

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
    plt.show()





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

