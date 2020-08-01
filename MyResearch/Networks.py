import torch.nn as nn



class model:
    def initialize(self,opt):
        self.opt=opt
        self.gpu_ids=opt.gpu_ids
        self.isTrain=opt.isTrain
        self.Tensor=torch.cuda.FloatTensor # Assuming we have a cuda GPU
        #Still need to set the save directory
        
        # Initialize the data structures to hold all each type of image
        batch_size=opt.batch_size
        new_dim=opt.fineSize
        self.input_A=self.Tensor(batch_size,3,new_dim,new_dim)
        self.input_B=self.Tensor(batch_size,3,new_dim,new_dim)
        self.input_img=self.Tensor(batch_size,3,new_dim,new_dim)
        self.input_A_gray=self.Tensor(batch_size,1,new_dim,new_dim)
        
        self.vgg_loss=PerceptualLoss(opt)
        self.vgg_loss.cuda()#--> Shift to the GPU
        

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
        
        
        
        
        
    