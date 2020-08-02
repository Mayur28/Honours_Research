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

class model:
    def initialize(self,opt):
        
        self.opt=opt
        self.gpu_ids=opt.gpu_ids
        self.isTrain=opt.isTrain
        self.Tensor=torch.cuda.FloatTensor # Assuming we have a cuda GPU
        #Still need to set the save directory
        
        # Initialize the data structures to hold all each type of image
        batch_size=opt.batch_size
        new_dim=opt.crop_size
        self.input_A=self.Tensor(batch_size,3,new_dim,new_dim)#We are basically creating a tensor to store 16 low-light colour images with size fineSize x fineSize
        self.input_B=self.Tensor(batch_size,3,new_dim,new_dim)# Same as above but now for storing the normal-light images (NOT THE RESULT!)
        self.input_img=self.Tensor(batch_size,3,new_dim,new_dim)
        self.input_A_gray=self.Tensor(batch_size,1,new_dim,new_dim)# this is for the attention maps
        
        self.vgg_loss=PerceptualLoss(opt)
        self.vgg_loss.cuda()#--> Shift to the GPU
        
        self.vgg=load_vgg16(self.gpu_ids)#This is for data parallelism
        #Actually load the VGG model(THIS IS CRUCIAL!)... This is the weights that we had to manually add
        self.vgg.eval() # We call eval() when some layers within the self.vgg network behave differently during training and testing... This will not be trained (Its frozen!)!
			#The eval function is often used as a pair with the requires.grad or torch.no grad functions (which makse sense)
            
        for param in self.vgg.parameters():
                param.requires_grad = False# Verified! For all the weights in the VGG network, we do not want to be updating those weights, therefore, we save computation using the above!

		#G_A : Is our only generator
		#D_A : Is the Global Discriminator
		#D_P : Is the patch discriminator
        print("HELLO, The end has been reached")
        

class PerceptualLoss(nn.Module):
    def __init__(self,opt):
        super(PerceptualLoss,self).__init__()
        self.opt=opt
        self.instance_norm=nn.InstanceNorm2d(512,affine=False)
        #512 is the number of features
		#They mention this Instance normalization in the paper to stabilize training
        
    def compute_vgg_loss(self,vgg_network,image,target):
        print("I am in Compute VGG_loss:%d"%image.shape)
        image_vgg=vgg_preprocess(image,self.opt)
        target_vgg=vgg_preprocess(target,self.opt)
        # Check if there is a work around this!
        
        img_feature_map=vgg(image_vgg,self.opt)# Get the feature map of the input image
        target_feature_map=vgg(target_vgg,self.opt)# Get the feature of the target image
        
        return torch.mean((self.instance_norm(img_feature_map) - self.instance_norm(target_feature_map)) ** 2) # We are using this to stabilize training( as mentioned in the paper)
    

class VGG(nn.Module): # optimize this, There should surely be some variations to this.. Understand what is trying to be achieved and then determine how to go about achieving this!
    def __init__(self):
        super(VGG,self).__init__()
        
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
        
    def forward(self,X,opt):# What is X?
        #Check how and when this is called!
        h=F.relu(self.conv1_1(X),inplace=True)
        
        
        
        
        
        
        
        
        
        
        
def load_vgg16(gpu_ids):
    vgg = Vgg16()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load('vgg16.weight'))# Adding the weights to the model
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg
        
        
        
        
        
    