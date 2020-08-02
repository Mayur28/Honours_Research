import torch
import torch.nn as nn
import torch.nn.functional as F

#SPICE EVERYTHING UP, TOO SIMILIAR!!!
#Blend it with Mask Shadow GAN
# Find other papers as well
# Do with my own understanding!!!
#This is just temporary, be ruthless thereafter

# Check the story about the transformation needed for preprocessing ( If RGB-> BGR is really necessary)

def pad_tensor(input):# Just check what is the dimensions of this input
    # Spice this up, can surely be done better?
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom): # We are just removing the padding
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]# This makes sense



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
        # Wherever below varibales are needed, just access self.opt.gpu_ids instead!
        #self.gpu_ids=opt.gpu_ids
        #self.isTrain=opt.isTrain
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
        
        opt.skip=True#--> Not needed because this will be in the training setting?
        
        self.Gen=Unet_generator1(opt)
        
        if(self.isTrain):
            self.G_Disc=PatchGAN(opt,False) # This declaration should have a patch option because they have different number of layers each.
            self.L_Disc=PatchGAN(opt,True)
		#G_A : Is our only generator
		#D_A : Is the Global Discriminator
		#D_P : Is the patch discriminator
        
        

class Unet_generator1(nn.Module):
    def __init__(self,opt):
        super(Unet_generator1,self).__init__()
        
        self.opt=opt
        # They explicitly set self.skip=skip, doesnt seem necessary
        # Surely I only need one of the MaxPooling layers, they area pretty much identical!
        # These will be used to resize the attention map to fit the latent result at each upsampling step
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
        self.max_pool5 = nn.MaxPool2d(2)# OVer and above everything, these maxpools can be removed because each is not different from each other.
        
        
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
            
        self.deconv6=nn.Conv2d(256,128,3,padding=1)
        self.conv7_1=nn.Conv2d(256,128,3,padding=1)
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
            
            
            
        self.deconv7=nn.Conv2d(128,64,3,padding=1)
        self.conv8_1=nn.Conv2d(128,64,3,padding=1)
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
            
            
        self.deconv8=nn.Conv2d(64,32,3,padding=1)
        self.conv9_1=nn.Conv2d(64,32,3,padding=1)
        self.LRelu9_1=nn.LeakyReLU(0.2,inplace=True)
        if(self.opt.norm_type=='batch'):
            self.norm9_1=nn.BatchNorm2d(32)
        else:
            self.norm9_1=nn.InstanceNorm2d(32)
        self.conv9_2=nn.Conv2d(32,32,3,padding=1)
        self.LRelu9_2=nn.LeakyReLU(0.2,inplace=True)
        
        self.conv10=nn.Conv2d(32,3,1) # This apparently has something to do with producing the latent space.
        
        self.tanh= nn.Tanh()# In the provided training conf., tanh is not used. But how do we ensure that the output is within an acceptable range?
        print("End of the generator")   
    
    def forward(self,input,gray):
        flag=0
        if input.size()[3] >2200:# This seems ridiculous! Test performance when this is removed
            avg=nn.avgPool2d(2)
            input=avg(input)
            gray=avg(gray)
            flag=1#--> Indicates that at the end, we need to upsample
            # Before Performing a forward pass on the tensor, we first pad the tensor containing the real (low-light) images
			#If the dimensions of the images are perfectly divisible by 16, we dont pad.
			# Otherwise, we pad the dimensions that are skew by the amount such that the dim. of the new padded version is divisible by 16.
			#The pad_tensor function performs the padding (if necessary) and returns how much padding was applied to each side which makes it easier when removing the padding later.
            
            input, pad_left,pad_right,pad_top,pad_bottom=pad_tensor(input)
            gray, pad_left, pad_right,pad_top, pad_bottom=pad_tensor(gray)
            
            # First downsample the attention map for all stages
            gray_2=self.downsample_1(gray)
            gray_3=self.downsample_2(gray_2)
            gray_4=self.downsample_3(gray_3)
            gray_5=self.downsample_4(gray_4)
            
            #print("Gray_2 size: %s" % str(gray_2.size()))
            #print("Gray_3 size: %s" % str(gray_3.size()))
            #print("Gray_4 size: %s" % str(gray_4.size()))
            #print("Gray_5 size: %s" % str(gray_5.size()))
            
            #Surely below can be automated!!!, do right at the end when I know what I'm doing!
            x=self.norm1_1(self.LRelu1_1(self.conv1_1(torch.cat((input,gray),1))))
            
            conv1=self.self.norm1_2(self.LRelu1_2(self.conv1_2(x)))
            x=self.max_pool1(conv1)
            
            x=self.norm2_1(self.LRelu2_1(self.conv2_1(x)))
            conv2=self.norm2_2(self.LRelu2_2(self.conv2_2(x)))
            x=self.max_pool2(conv2)
            
            x=self.norm3_1(self.LRelu3_1(self.conv3_1(x)))
            conv3=self.norm3_2(self.LRelu3_2(self.conv3_2(x)))
            x=self.max_pool3(conv3)
            
            x=self.norm4_1(self.LRelu4_1(self.conv4_1(x)))
            conv2=self.norm4_2(self.LRelu4_2(self.conv4_2(x)))
            x=self.max_pool4(conv2)
            
            x=self.norm5_1(self.LRelu5_1(self.conv5_1(x)))
            x=x*gray_5
            conv_5=self.norm5_2(self.LRelu5_2(self.conv5_2(x)))
            
            #Bottleneck has been reached( I think, but then, why is the att map already being multiplied?) - start upsampling
			# Experiment here to see if bilinear upsampling really is this best option.
            
            conv5=F.upsample(conv5,scale_factor=2,mode='bilinear')
            conv4=conv4*gray_4
            up6=torch.cat([self.deconv5(conv5),conv4],1)
            x=self.norm6_1(self.self.LRelu6_1(self.conv6_1(up6)))
            conv6=self.norm6_2(self.LRelu6_2(self.conv6_2(x)))
            
            conv6=F.upsample(conv6,scale_factor=2,mode='bilinear')
            conv3=conv3*gray_3
            up7=torch.cat([self.deconv6(conv6),conv3],1)
            x=self.norm7_1(self.self.LRelu7_1(self.conv7_1(up7)))
            conv7=self.norm7_2(self.LRelu7_2(self.conv7_2(x)))
            
            conv7=F.upsample(conv7,scale_factor=2,mode='bilinear')
            conv2=conv2*gray_2
            up8=torch.cat([self.deconv7(conv7),conv2],1)
            x=self.norm8_1(self.self.LRelu8_1(self.conv8_1(up8)))
            conv8=self.norm8_2(self.LRelu8_2(self.conv8_2(x)))
            
            conv8=F.upsample(conv8,scale_factor=2,mode='bilinear')
            conv1=conv1*gray
            up9=torch.cat([self.deconv8(conv8),conv1],1)
            x=self.norm9_1(self.self.LRelu9_1(self.conv9_1(up9)))
            conv9=self.norm9_2(self.LRelu9_2(self.conv9_2(x)))
            
            latent = self.conv10(conv9)# What is this for?

            if self.opt.times_residual:# True!
                latent = latent*gray

            if self.opt.tanh:
                latent = self.tanh(latent)# Oddly does not apply to us
            if self.skip:
            	output = latent + input*self.opt.skip# This is a breakthrough! The latent result is added to the low-light image to form the output.

            
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1: # If fineSize>2200 which resulting in having to perform AvgPooling
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent # Want to see what is this latent!
        else:
            return output
            

class PatchGAN(nn.Module):
    def __init__(self,opt):
        super(NoNormDiscriminator, self).__init__()
        
        self.opt=opt
        
        sequence=[nn.Conv2d(3,64,kernel_size=4,stride=2,padding=2,nn.LeakyReLU(0.2,True))]
        
        # The rubbish below can be modified!
        #Filter out this rubbish
        # Look at flagship papers, as well as morphing EGAN's alternate implementations and get ideas from other papers
        
        nf_mult=1
        nf_mult_prev=1
        for n in range(1,self.opt.n_layers_D):
            nf_mult_prev=nf_mult
            nf_mult=min(2**n,8)
            sequence+=[nn.Conv2d(ndf*nf*mult_prev,ndf*nf_mult,kernel_size=4,stride=2,padding=2),
            nn.LeakyReLU(0.2,True)]
            
        nf_mult_prev=nf_mult
        nf_mult= min(2*self.opt.n_layers_D,8)
        sequence+=[nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=4,stride=1,padding=2),nn.LeakyReLU(0.2,True)]
        
        sequence+=[nn.Conv2d(ndf*nf_mult,1,kernel_size=4,stride=1,padding=2)]
        
        # Reda up on the story about the sigmoid at the end. If yes, += it here
        
        self.model=nn.Sequential(sequence)
        
    def forward(self,input):
        return self.model(input)#<-- pass through the discriminator itself which is represented by self.model
            
        
            
            
            
        
        
    
        

            
        
        
        
        
        
        
        
        

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
        
        
        
        
        
    