import torch
import torch.nn as nn
import torch.nn.functional as F

#SPICE EVERYTHING UP, TOO SIMILIAR!!!
#Blend it with Mask Shadow GAN
# Find other papers as well
# Do with my own understanding!!!
#This is just temporary, be ruthless thereafter

# Check the story about the transformation needed for preprocessing ( If RGB-> BGR is really necessary)

def weight_init(model):
    class_name=model.__class__.__name__
    if class_name.find('Conv')!=-1:
        torch.nn.init.normal_(model.weight.data,0.0,0.02)
    elif class_name.find('BatchNorm2d')!=-1:
        torch.nn.init.normal_(model.weight.data,1.0,0.02)
        torch.nn.init.constant_(model.bias.data,0.0)

class The_Model:
    def __init__(self,opt):
        
        self.opt=opt
        self.gpu_ids=opt.gpu_ids
        self.isTrain=opt.isTrain
        #Still need to set the save directory
        
        # Initialize the data structures to hold all each type of image
        batch_size=opt.batch_size
        new_dim=opt.crop_size
        #I'm assuming that a CUDA GPU is used.
        self.input_A=torch.cuda.FloatTensor(batch_size,3,new_dim,new_dim)#We are basically creating a tensor to store 16 low-light colour images with size crop_size x crop_size
        self.input_B=torch.cuda.FloatTensor(batch_size,3,new_dim,new_dim)# Same as above but now for storing the normal-light images (NOT THE RESULT!)
        self.input_img=torch.cuda.FloatTensor(batch_size,3,new_dim,new_dim)
        self.input_A_gray=torch.cuda.FloatTensor(batch_size,1,new_dim,new_dim)# this is for the attention maps
        
        self.vgg_loss=PerceptualLoss(opt) # Use the alternate implementation when experimenting
        self.vgg_loss.cuda()#--> Shift to the GPU
        
        self.vgg=load_vgg(self.gpu_ids)#This is for data parallelism
        #Actually load the VGG model(THIS IS CRUCIAL!)
        self.vgg.eval() # We call eval() when some layers within the self.vgg network behave differently during training and testing... This will not be trained (Its frozen!)!
        #The eval function is often used as a pair with the requires.grad or torch.no grad functions (which makes sense)
        #I'm setting it to eval() because it's not being trained in anyway
            
        for weights in self.vgg.parameters():# THIS IS THE BEST WHY OF DOING THIS
                weights.requires_grad = False# Verified! For all the weights in the VGG network, we do not want to be updating those weights, therefore, we save computation using the above!
        
        opt.skip=True
        
        self.Gen=Unet_generator1(opt)
        
		#G_A : Is our only generator
		#D_A : Is the Global Discriminator
		#D_P : Is the patch discriminator
        
        

class Unet_generator1(nn.Module):
    def __init__(self,opt):
        super(Unet_generator1,self).__init__()
        
        self.opt=opt
        print("SKIP: %s"%self.opt.skip)
        # They explicitly set self.skip=skip, doesnt seem necessary
        
        # This will be used to resize the attention map to fit the latent result at each upsampling step
        self.resized_att1 = nn.MaxPool2d(2)# This is seperate( this is for the attention maps to fit the size of the filters in each layer) -----> This is for downsampling the attention map. At each step, the size of the attention map is halved.
        self.resized_att2 = nn.MaxPool2d(2)
        self.resized_att3 = nn.MaxPool2d(2)
        self.resized_att4 = nn.MaxPool2d(2)
        
        self.conv1_1=nn.Conv2d(4,32,3,padding=1)# 4 because of the the RGB image and the attention map...
        self.LRelu1_1=nn.LeakyReLU(0.2,inplace=True) # Inplace is to make the changes directly without producing additional output (it is what it says it is)
        if(self.opt.norm_type=='Batch'):
            self.norm1_1=nn.BatchNorm2d(32)
        else:
            self.norm1_1=nn.InstanceNorm2d(32)
            
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='Batch'):
            self.norm1_2 = nn.BatchNorm2d(32)
        else:
            self.norm1_2=nn.InstanceNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)# Try to get rid of this form of downsampling (Read Radford)
        
        
        
        self.conv2_1=nn.Conv2d(32,64,3,padding=1)
        self.LRelu2_1=nn.LeakyReLU(0.2,inplace=True)
        if(self.opt.norm_type=='Batch'):
            self.norm2_1=nn.BatchNorm2d(64)
        else:
            self.norm2_1=nn.InstanceNorm2d(64)
            
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='Batch'):
            self.norm2_2 = nn.BatchNorm2d(64)
        else:
            self.norm2_2=nn.InstanceNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)
        
        
        self.conv3_1=nn.Conv2d(64,128,3,padding=1)
        self.LRelu3_1=nn.LeakyReLU(0.2,inplace=True)
        if(self.opt.norm_type=='Batch'):
            self.norm3_1=nn.BatchNorm2d(128)
        else:
            self.norm3_1=nn.InstanceNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='Batch'):
            self.norm3_2 = nn.BatchNorm2d(128)
        else:
            self.norm3_2=nn.InstanceNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)
        
        
        self.conv4_1=nn.Conv2d(128,256,3,padding=1)
        self.LRelu4_1=nn.LeakyReLU(0.2,inplace=True)
        if(opt.norm_type=='Batch'):
            self.norm4_1=nn.BatchNorm2d(256)
        else:
            self.norm4_1=nn.InstanceNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if (opt.norm_type=='Batch'):
            self.norm4_2 = nn.BatchNorm2d(256)
        else:
            self.norm4_2=nn.InstanceNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)
        
        
        
        self.conv5_1=nn.Conv2d(256,512,3,padding=1)
        self.LRelu5_1=nn.LeakyReLU(0.2,inplace=True)
        if(opt.norm_type=='Batch'):
            self.norm5_1=nn.BatchNorm2d(512)
        else:
            self.norm5_1=nn.InstanceNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if (opt.norm_type=='Batch'):
            self.norm5_2 = nn.BatchNorm2d(512)
        else:
            self.norm5_2=nn.InstanceNorm2d(512)
        self.max_pool5 = nn.MaxPool2d(2)
        
        
        #The bottleneck has been reached, we now enter the decoder. We need to now upsample to produce the sample.
        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)# IT SEEMS THAT THEY ALREADY ATTEMPTED WHAT I WANTED TO DO( USE THE TRANSPOSE CONV LAYER TO UPSAMPLE)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=1)#This is apparently referred to as a bilinear upsampling layer.(According to the paper). This apparently gets rid of checkerboard effects
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1)# Try to get an intuition of how the no. of filters,kernel_size and strides are configured to achieve different characteristics
        self.LRelu6_1 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='batch'):
            self.norm6_1 =  nn.BatchNorm2d(256)
        else:
            self.norm6_1= nn.InstanceNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.LRelu6_2 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='batch'):
            self.norm6_2 =  nn.BatchNorm2d(256)
        else:
            self.norm6_2 = nn.InstanceNorm2d(256)
            
        self.deconv6=nn.Conv2d(256,128,padding=1)
        self.conv7_1=nn.Conv2d(256,128,padding=1)
        self.LRelu7_1=nn.LeakyReLU(0.2,inplace=True)
        if (self.opt.norm_type=='batch'):
            self.norm7_1 =  nn.BatchNorm2d(128)
        else:
            self.norm7_1= nn.InstanceNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.LRelu7_2 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='batch'):
            self.norm7_2 =  nn.BatchNorm2d(128)
        else:
            self.norm7_2=nn.InstanceNorm2d(128)
            
            
            
        self.deconv7=nn.Conv2d(128,64,padding=1)
        self.conv8_1=nn.Conv2d(128,64,padding=1)
        self.LRelu8_1=nn.LeakyReLU(0.2,inplace=True)
        if (self.opt.norm_type=='batch'):
            self.norm8_1 =  nn.BatchNorm2d(64)
        else:
            self.norm8_1= nn.InstanceNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.LRelu8_2 = nn.LeakyReLU(0.2, inplace=True)
        if (self.opt.norm_type=='batch'):
            self.norm8_2 =  nn.BatchNorm2d(64)
        else:
            self.norm8_2=nn.InstanceNorm2d(64)
            
            
        self.deconv8=nn.Conv2d(64,32,padding=1)
        self.conv9_1=nn.Conv2d(64,32,padding=1)
        self.LRelu9_1=nn.LeakyReLU(0.2,inplace=True)
        if(self.opt.norm_type=='batch'):
            self.norm9_1=nn.BatchNorm2d(32)
        else:
            self.norm9_1=nn.InstanceNorm2d(32)
        seld.conv9_2=nn.Conv2d(32,32,3,padding=1)
        self.LRelu9_2=nn.LeakyReLU(0.2,inplace=True)
        
        self.conv10=nn,Conv2d(32,3,1) # This apparently has something to do with producing the latent space.
        
        if self.opt.tanh:
            self.tanh= nn.Tanh()# In the provided training conf., tanh is not used. But how do we ensure that the output is within an acceptable range?
            
        

            
        
        
        
        
        
        
        
        

class PerceptualLoss(nn.Module):# All NN's needed to be based on this class and have a forward() function
    def __init__(self,opt):
        super(PerceptualLoss,self).__init__()
        self.opt=opt
        self.instance_norm=nn.InstanceNorm2d(512,affine=False)# <-- Affine determines if some of the parameters for normalizing are "learnable" or not.
        #512 is the number of features
		#This is to stabilize training
        
    def compute_vgg_loss(self,vgg_network,image,target):
        print("I am in Compute VGG_loss:%d"%image.shape)
        image_vgg=cv2.normalize(image,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#vgg_preprocess(image,self.opt)--> This function was supposed to convert the RGB image to BGR and convert the normalized image [-1,1] from the tanh function to [0,255]... Im removing it now, but check ifit is really necessary to change the range.
        target_vgg=cv2.normalize(target,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#vgg_preprocess(target,self.opt)
        # Check if there is a work around this!
        
        img_feature_map=vgg(image_vgg,self.opt)# Get the feature map of the input image
        target_feature_map=vgg(target_vgg,self.opt)# Get the feature of the target image
        
        return torch.mean((self.instance_norm(img_feature_map) - self.instance_norm(target_feature_map)) ** 2)# --> According to the provided function
    

class Vgg(nn.Module): # optimize this, There should surely be some variations to this.. Understand what is trying to be achieved and then determine how to go about achieving this!
    
    
    def __init__(self):
        super(Vgg,self).__init__()
        
        
        #This needs to be changed!
        self.conv1_1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        
        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        
        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.conv3_3=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        
        self.conv4_1=nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.conv4_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        
        self.conv5_1=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.conv5_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        self.conv5_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        
    def forward(self,input,opt):# What is X?
        
        # Alot over variation can come out of this function
        #Check how and when this is called!
        act=F.relu(self.conv1_1(input),inplace=True)
        act=F.relu(self.conv1_2(act),inplace=True)
        act=F.max_pool2d(act,kernel_size=2,stride=2)
        
        act=F.relu(self.conv2_1(act),inplace=True)
        act=F.relu(self.conv2_2(act),inplace=True)
        act=F.max_pool2d(act,kernel_size=2,stride=2)
        
        act=F.relu(self.conv3_1(act),inplace=True)
        act=F.relu(self.conv3_2(act),inplace=True)
        act=F.relu(self.conv3_3(act),inplace=True)
        act=F.max_pool2d(act, kernel_size=2, stride=2)

        act = F.relu(self.conv4_1(act), inplace=True)
        act = F.relu(self.conv4_2(act), inplace=True)
        act = F.relu(self.conv4_3(act), inplace=True)

        relu5_1 = F.relu(self.conv5_1(act), inplace=True)
        return relu5_1
  
        
        

def load_vgg(gpu_ids):
    vgg = Vgg()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load('vgg16.weight'))# Adding the weights to the model
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg
        
        
        
        
        
    