import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob

from torch import nn
import os.path
import torchvision.transforms as transforms
import PIL
from pdb import set_trace as st


def DataLoader(opt):
    data_loader=DataLoader(opt)
    return data_loader


def import_dataset(directory):
    images = []
    for filename in glob.glob(directory+str("/*.png")) or glob.glob(directory+str("/*.jpg")) : # I'm only allowing png and jpg images as training images
        im=Image.open(filename).convert('RGB')
        images.append(im)
    return images

def config_transforms(opt):
    trans_list=[]
    # For data augmentation, perform random cropping, sometimes horizontal flipping, sometimes vertical flipping and finalize normalize( to range [-1,1])
    trans_list+=[transforms.RandomCrop(opt.crop_size),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.RandomVerticalFlip(p=0.35),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))] # Get the image to [-1,1]
    return transforms.Compose(trans_list)

def the_gray_transform():
    trans=[transforms.ToPILImage(),transforms.Grayscale(1),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    return transforms.Compose(trans)

class DataLoader:
    def __init__(self,opt):
        self.opt=opt
        self.dataset=FullDataset(opt)# Remember that self.dataset needs to have inherited from the built-in Dataset class to be used below... pin_memory apparently has to do with making it faster to load data to the gpu
        self.dataloader= torch.utils.data.DataLoader(self.dataset,batch_size=opt.batch_size,shuffle= True, pin_memory=True,num_workers=6)

    def load(self):# This will return the iterable over the dataset
        return self.dataloader

    #This function is compulsory when creating custom dataloaders!
    def __len__(self):
        return len(self.dataset)


class FullDataset(data.Dataset):
    # This class definitely needs to overwrite __len__ and  __getitem__
    def __init__(self,opt):
        super(FullDataset, self).__init__()
        self.opt=opt
        #Form the path's to the data
        A_directory=os.path.join('../final_dataset',opt.phase+'A')
        B_directory=os.path.join('../final_dataset',opt.phase+'B')


        self.A_imgs = import_dataset(A_directory)
        self.B_imgs = import_dataset(B_directory)


        self.A_size=len(self.A_imgs)
        self.B_size=len(self.B_imgs)
        self.transform=config_transforms(opt)
        self.gray_transform=the_gray_transform()

    def __getitem__(self,index):
        A_img=self.A_imgs[index%self.A_size]# To avoid going out of bounds
        B_img=self.B_imgs[index% self.B_size]

        A_img=self.transform(A_img)#This is where we actually perform the transformation. These are now tensors that are normalized
        B_img=self.transform(B_img)


        input_img=A_img
        # We are going from a normal 600x400 image ( In the PIL format),
        #after the transform, the image is manipulated and converted into a tensor for each image ( resulting size=[3,320,320])
        r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
        A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img':input_img}



    def __len__(self):
        return max(self.A_size,self.B_size)

# Right now, this is directly copied over, NEEDS TO BE REDONE URGENTLY!!!
# Improve the implementation here( vast amount of rom from improvement here!)
# Look at VGG preprocess on how to convert from [-1,1] to [0,255]
def TensorToImage(img_tensor,imtype=np.uint8):
    usable_images=img_tensor[0].cpu().float().numpy()
    usable_images=(np.transpose(usable_images, (1, 2, 0)) + 1) / 2.0 * 255.0
    usable_images=np.clip(usable_images,0,255)
    return usable_images.astype(imtype)
