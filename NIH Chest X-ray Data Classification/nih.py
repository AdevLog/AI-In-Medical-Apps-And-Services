# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 07:59:50 2022

@author: R6
"""
import torch
# import torch.nn.functional as F
# import torchvision

from torchvision import models, transforms
# from torch.autograd import Variable
# from torchcam.utils import overlay_mask
# from torchvision.io.image import read_image
# from torchvision.transforms.functional import normalize, resize, to_pil_image
# from torchcam.methods import GradCAM
# import tensorflow as tf

import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize(
                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# model = models.densenet121(pretrained=False)
# # Checkpoint = torch.load("1591626098-chesXnet.pt")
# model.load_state_dict(torch.load("1591626098-chesXnet.pt"), strict=False)
# model.eval()
# model = torch.load("1591626098-chesXnet.pt")

def get_prediction(img_bytes):
    tensor = transform_image(img_bytes)
    tensor = tensor.to(device)
    # output = model.forward(tensor)
    output = model(tensor)
     
    probs = torch.nn.functional.softmax(output, dim=1)
    val, idx = torch.max(probs, 1)
    # print("probs",probs)
    return val, idx


# outputs = model(image)
# probs = outputs.cpu().data.numpy()

def plot_loss():
    
    columns = ["epoch","train_loss","val_loss"]
    df = pd.read_csv("log_train9.csv", usecols=columns)
    plt.plot(df.epoch, df.train_loss, color = 'g', linewidth = 0.8, label = "train_loss")
    plt.plot(df.epoch, df.val_loss, color = 'r', linewidth = 0.8, label = "val_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train and val loss(data: 112121)')
    
    plt.legend()
    plt.show()
    
    df = pd.read_csv("log_train5.csv", usecols=columns)
    plt.plot(df.epoch, df.train_loss, color = 'g', linewidth = 0.8, label = "train_loss")
    plt.plot(df.epoch, df.val_loss, color = 'r', linewidth = 0.8, label = "val_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train and val loss(data: 19708)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    model = models.densenet121(pretrained=False, num_classes=2)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load("model_state_dict_4.pt"), strict=False)
    image_path="00000013_011.png"
    image = plt.imread(image_path)
    image = np.stack((image,)*3, axis=-1)
    # plt.imshow(image)
    print("Is "+image_path+" Mass or not?")
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        decode = np.stack((image_bytes,)*3, axis=-1)
     
        conf, y_pre = get_prediction(decode)
        # # img_pil = Image.open("/home/alejandro/workspace/uav_detection/images/" + str(i + 1) + ".jpg")
        # image_path="00000005_005.png"
        # img_pil = Image.open(image_path)
        # img_tensor = my_transforms(img_pil ).float()
        # img_tensor = img_tensor.unsqueeze_(0)
        # img_tensor = img_tensor.to(device)
        
        # fc_out = model(Variable(img_tensor))
        
    
        # # output = fc_out.detach().cpu().numpy()
        # output = fc_out.cpu().data.numpy()
        # print(output.argmax())
        # print(output)
        
        if y_pre[0].item() == 0:
            imgClasses = 'Mass'
        else:
            imgClasses = 'Not Mass'        
        print(imgClasses+ ' at confidence score: %f'%(conf[0].item()))
        
    plot_loss()