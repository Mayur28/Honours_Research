import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import glob


def DataLoader(opt):
    data_loader = DataLoader(opt)
    return data_loader


def import_dataset(directory):
    images = []
    for filename in glob.glob(directory + str("/*.png")) or glob.glob(directory + str("/*.jpg")):  # I'm only allowing png and jpg images as training images
        im = Image.open(filename).convert('RGB')
        images.append(im)
    return images


def config_transforms(opt):
    trans_list = []
    # For data augmentation, perform random cropping, sometimes horizontal flipping, sometimes vertical flipping and finalize normalize( to range [-1,1])
    if opt.phase == 'train':
        trans_list += [transforms.Resize((512,512)),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomVerticalFlip(p=0.5),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # Get the image to [-1,1]
    else:
        trans_list += [
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # Get the image to [-1,1]

    return transforms.Compose(trans_list)

def gray_transform():
    trans_list = []
    trans_list += [transforms.ToPILImage(),
                   transforms.Grayscale(num_output_channels =1),
                   transforms.ToTensor()]
    return transforms.Compose(trans_list)



class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = FullDataset(opt)  # Remember that self.dataset needs to have inherited from the built-in Dataset class to be used below... pin_memory apparently has to do with making it faster to load data to the gpu
        if opt.phase == 'train':
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=6)
        else:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    def load(self):  # This will return the iterable over the dataset
        return self.dataloader

    # This function is compulsory when creating custom dataloaders!
    def __len__(self):
        return len(self.dataset)


class FullDataset(data.Dataset):
    # This class definitely needs to overwrite __len__ and  __getitem__
    def __init__(self, opt):
        super(FullDataset, self).__init__()
        self.opt = opt
        # Form the path's to the data
        A_directory = os.path.join(opt.data_source, opt.phase + 'A')
        B_directory = os.path.join(opt.data_source, opt.phase + 'B')

        self.A_imgs = import_dataset(A_directory)
        if (opt.phase == 'train'):
            self.B_imgs = import_dataset(B_directory)
        else:
            self.B_imgs = self.A_imgs  # We just need some image so that this portion
            # of the code can be reused for training and testing
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        self.transform = config_transforms(opt)
        self.gray_transform = gray_transform()

    def __getitem__(self, index):
        A_img = self.A_imgs[index % self.A_size]  # To avoid going out of bounds
        B_img = self.B_imgs[index % self.B_size]

        width, height = A_img.size
        A_img = self.transform(A_img)  # This is where we actually perform the transformation. These are now tensors that are normalized
        B_img = self.transform(B_img)

        # We are going from a normal 600x400 image ( In the PIL format),
        # after the transform, the image is manipulated and converted into a tensor for each image ( resulting size=[3,320,320])

        #A_gray = 1 - self.gray_transform(A_img)

        r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.  # This is definitely the best way... My way only worked for an older version of torch
        A_gray = torch.unsqueeze(A_gray, 0)
        #print(A_gray.size())
        # Size of A_gry is [1,340,340]
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray}

    def __len__(self):
        return max(self.A_size, self.B_size)


def TensorToImage(img_tensor):
    for_disp = img_tensor[0].cpu().float().numpy()
    for_disp = (np.transpose(for_disp, (1, 2, 0)) + 1) / 2.0 * 255.0
    for_disp = np.clip(for_disp, 0, 255)
    return for_disp.astype(np.uint8)
