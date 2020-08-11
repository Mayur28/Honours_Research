import os
import torch
import torch.utils.data as data
import cv2
import imghdr
import torchvision.transforms as transforms# Try to remove
#I'm trying without handling the image path! Looks like this will need special attention when saving the images( This will come much later)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#Try to go the opencv route

def DataLoader(opt):
    data_loader=DataLoader(opt)
    return data_loader

def MakeDataset(opt):
    dataset=FullDataset(opt)
    return dataset

def import_dataset(directory):
    images = []# This seems to  be the way to go...
    for root, _, files in sorted(os.walk(directory)):
        for file_name in files:
            if  imghdr.what(directory+"/"+file_name)=='png' or  imghdr.what(directory+"/"+file_name)=='jpeg':
                path = os.path.join(root, file_name)
                img =Image.open(path).convert('RGB')# cv2.imread(path,1)# Will be read in BGR!
                images.append(img)
    return images

def config_transforms(opt):# I Should account for other kinds of transforms such as resize and crop (THIS IS WHERE MY DATA AUGMENTATION WILL GO. RIGHT NOW, IM ONLY DOING RANDOM CROPPING)
    trans_list=[]
    trans_list.append(transforms.RandomCrop(opt.crop_size))

    #To normalize the data to the range [-1,1]
    trans_list+=[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]# this is neat. Looking at one channel (column), we are specifying mean=std=0.5 which normalizes the images to [-1,1]
    return transforms.Compose(trans_list)

class DataLoader:
    def __init__(self,opt):
        self.opt=opt
        self.dataset=MakeDataset(opt)
        self.dataloader= torch.utils.data.DataLoader(self.dataset,batch_size=opt.batch_size,shuffle= True, num_workers=6)

    def load(self):# This will return the iterable over the dataset
        return self.dataloader

    #This function is compulsory when creating custom dataloaders!
    def __len__(self):
        return len(self.dataset)

    #Try to implement a __getitem__ to extract individual instances as well!


class FullDataset(data.Dataset):# I've inherited what I had to
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

    def __getitem__(self,index):
        A_img=self.A_imgs[index%self.A_size]# To avoid going out of bounds
        B_img=self.B_imgs[index% self.B_size]
        #A_path=self.A_paths[index% self.A_size]
        #B_path=self.B_paths[index% self.B_size]
        A_img=self.transform(A_img)#This is where we actually perform the transformation
        B_img=self.transform(B_img)

        input_img=A_img
        #A_gray=cv2.cvtColor(input_img,0)
        #A_gray=torch.unsqueeze(A_gray,0)
        r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. #Verified: The weird numbers are for going from RGB to grayscale
        A_gray = torch.unsqueeze(A_gray, 0)#Returns a new tensor with a


        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img}

    def __len__(self):
        return max(self.A_size,self.B_size)

# Right now, this is directly copied over, NEEDS TO BE REDONE URGENTLY!!!
# Improve the implementation here( vast amount of rom from improvement here!)
def TensorToImage(img_tensor,imtype=np.uint8):
    usable_images=img_tensor[0].cpu().float().numpy()
    usable_images=(np.transpose(usable_images, (1, 2, 0)) + 1) / 2.0 * 255.0
    usable_images=np.maximum(usable_images,0)
    usable_images=np.minimum(usable_images,255)
    return usable_images.astype(imtype)

# Find out exactly what is going on here( detaching and manipulating???)
def AttentionToImage(img_tensor,imtype=np.uint8):
    tensor=img_tensor[0]
    tensor=torch.cat((tensor, tensor, tensor), 0)
    usable_images=tensor.cpu().float().numpy()
    usable_images = (np.transpose(usable_images, (1, 2, 0))) * 255.0
    usable_images = usable_images/(usable_images.max()/255.0)
    return usable_images.astype(imtype)

def LatentToImage(img_tensor,imtype=np.uint8):
    usable_images = img_tensor[0].cpu().float().numpy()
    usable_images = (np.transpose(usable_images, (1, 2, 0))) * 255.0
    usable_images = np.maximum(usable_images, 0)
    usable_images = np.minimum(usable_images, 255)
    return usable_images.astype(imtype)
