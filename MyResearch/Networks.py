import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from ManageData import TensorToImage
import os
from collections import OrderedDict
import functools
import numpy as np
import glob


# Check what would be the story when testing? Would the networks below be created( sound silly but isnt!)

def weights_init(model): # This is optimized!!!
    class_name=model.__class__.__name__
    if class_name.find('Conv')!=-1:
        torch.nn.init.normal_(model.weight.data,0.0,0.02)
    elif class_name.find('BatchNorm2d')!=-1:
        torch.nn.init.normal_(model.weight.data,1.0,0.02)
        torch.nn.init.constant_(model.bias.data,0.0)


def add_padding(input):# Optimized!
    height, width= input.shape[2],input.shape[3]

    optimal_size=512
    pad_left = pad_right = pad_top= pad_bottom= 0
    if(width!=optimal_size):
        width_diff= optimal_size-width
        pad_left= int(np.ceil(width_diff/2))
        pad_right= width_diff-pad_left
    if(height!=optimal_size):
        height_diff=optimal_size-height
        pad_top= int(np.ceil(height_diff/2))
        pad_bottom= height_diff-pad_top

    padding= nn.ReflectionPad2d((pad_left,pad_right,pad_top,pad_bottom))
    input=padding(input)
    return input,pad_left,pad_right,pad_top,pad_bottom

def remove_padding( input,pad_left,pad_right,pad_top,pad_bottom):
    height,width =input.shape[2],input.shape[3]
    return input[:,:,pad_top:height-pad_bottom,pad_left:width-pad_right]

class The_Model: # This is the grand model that encompasses everything ( the generator, both discriminators and the VGG network)
    def __init__(self,opt):

        self.opt=opt
        #I'm assuming that a CUDA GPU is used.
        self.input_A=torch.cuda.FloatTensor(opt.batch_size,3,opt.crop_size,opt.crop_size)#We are basically creating a tensor to store 16 low-light colour images with size crop_size x crop_size
        self.input_B=torch.cuda.FloatTensor(opt.batch_size,3,opt.crop_size,opt.crop_size)# Same as above but now for storing the normal-light images (NOT THE RESULT!)
        self.input_img=torch.cuda.FloatTensor(opt.batch_size,3,opt.crop_size,opt.crop_size)
        self.input_A_gray=torch.cuda.FloatTensor(opt.batch_size,1,opt.crop_size,opt.crop_size)# this is for the attention maps

        self.vgg_loss=PerceptualLoss()
        self.vgg_loss.cuda()#--> Shift to the GPU

        self.vgg=load_vgg(self.opt.gpu_ids)#This is for data parallelism
        self.vgg.eval() # We call eval() when some layers within the self.vgg network behave differently during training and testing... This will not be trained (Its frozen!)!
        #The eval function is often used as a pair with the requires.grad or torch.no grad functions (which makes sense)
        #I'm setting it to eval() because it's not being trained in anyway

        for weights in self.vgg.parameters():# THIS IS THE BEST WAY OF DOING THIS
            weights.requires_grad = False# Verified! For all the weights in the VGG network, we do not want to be updating those weights, therefore, we save computation using the above!

        # Above looks optimized
        # We shouldnt be coming here in the first place when we are testing, just load directly from the latest model that we saved
        self.Gen=make_G(opt)
        if self.opt.phase=='test':
            self.load_model(self.Gen,'Gener')# Just get the latest!
            #self.load_model(self.Gen,'Global_Disc')# Just get the latest!
            #self.load_model(self.Gen,'Local_Disc')# Just get the latest!


        if(self.opt.phase=='train'): # Why would we be instantiating new discriminators when we are testing?? We shouldnt be coming here in the first place.
            self.G_Disc=make_Disc(opt,False)
            self.L_Disc=make_Disc(opt,True)

            self.model_loss=GANLoss()

            #Check if the below optimizers are set in accordance with Radford!
            self.G_optimizer=torch.optim.Adam(self.Gen.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
            self.G_Disc_optimizer=torch.optim.Adam(self.G_Disc.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
            self.L_Disc_optimizer=torch.optim.Adam(self.L_Disc.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
            #if self.opt.phase!='train': # We shouldn't be coming here in the first place! We should directly be able to load the model...
            #   self.Gen.eval()# Do we really need this? I dont think that we are instantiating a new network when predicting, we're just loading an existing network...

    def forward(self):
        # Look into what the Variable stuff is for

        self.real_A=Variable(self.input_A)#Variable is basically a tensor (which represents a node in the comp. graph) and is part of the autograd package to easily compute gradients
        self.real_B=Variable(self.input_B) #This contains the normal-light images ( sort of our reference images)
        self.real_A_gray=Variable(self.input_A_gray) # This is the attention map
        self.real_img=Variable(self.input_img) #In our configuation, input_img=input_A

        #Make a prediction!
        # What is the latent used for?
        the_input=torch.cat([self.real_img,self.real_A_gray],1)
        self.fake_B= self.Gen.forward(the_input)# We forward prop. a batch at a time, not individual images in the batch!

    #Perfect

    def set_input(self,input):
        input_A=input['A']
        input_B=input['B']
        input_A_gray=input['A_gray']
        input_img=input['input_img']

        # Copy the data to there respective cuda Tensors used for training on the GPU
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_img.resize_(input_img.size()).copy_(input_img)

    def perform_update(self):  #Do the forward,backprop and update the weights... this is a very powerful and 'highly abstracted' function
        # forward
        # This is for optimizing the generator.
        self.forward()# This produces the fake samples and sets up some of the variables that we need ie. we initialize the fake patch and the list of patches. But why do we need the single patch and the list of patches? # NOTE! THIS DOES NOT PASS THROUGH THE NETWORK!!! EXPERIMENT THOROUGHLY HERE!
        self.G_optimizer.zero_grad()# Check the positioning of this statement (can it be first?)
        self.backward_G()
        self.G_optimizer.step()

        # Now onto updating the discriminator!
        self.G_Disc_optimizer.zero_grad()
        self.backward_G_Disc()
        self.L_Disc_optimizer.zero_grad()
        self.backward_L_Disc()
        self.G_Disc_optimizer.step()
        self.L_Disc_optimizer.step()

    def predict(self):
        # Why do we need these here?
        self.real_A= Variable(self.input_A)
        self.real_A.requires_grad=False
        self.real_A_gray= Variable(self.input_A_gray)
        self.real_A_gray.requires_grad=False
        the_input= torch.cat([self.real_A,self.input_A_gray],1)
        self.fake_B = self.Gen.forward(the_input)


    def backward_G(self):
        # First let the discriminator make a prediction on the fake samples
        #This is the part recommended by Radford where we test real and fake samples in stages
        pred_fake=self.G_Disc.forward(self.fake_B)
        # torch.mean() is a scalar... what is going on?
        pred_real=self.G_Disc.forward(self.real_B)

        # The switching is now clarified as explained in the paper
        self.Gen_adv_loss= (self.model_loss(pred_real  - torch.mean(pred_fake), False) + self.model_loss(pred_fake  - torch.mean(pred_real), True)) / 2
        # In a seperate variable, we start accumulating the loss from the different aspects (which include the loss on the patches and the vgg loss)


        # Experiment as much as possible with the latent variable and understand what exactly does it represent. Find a better way of doing the cropping, their approach looks lame...
        w=self.real_A.size(3)
        h=self.real_B.size(2)

        # Check if there is really a need for these seperate patches
        # fake_B is a tensor of many images, how do we know from which image in the tensor are we cropping from? It seems that we take a patch from each image in the tensor (containing 16 images each)

        self.fake_patch_list=[]
        self.real_patch_list=[]
        self.input_patch_list=[]

        # This will basically create 8 batches (of 16 patches each)
        for i in range(self.opt.patchD_3):

            w_offset=random.randint(0,max(0,w-self.opt.patch_size-1))
            h_offset=random.randint(0,max(0,h-self.opt.patch_size-1))

            self.fake_patch_list.append(self.fake_B[:,:,h_offset:h_offset+self.opt.patch_size,w_offset:w_offset+self.opt.patch_size])
            self.real_patch_list.append(self.real_B[:,:,h_offset:h_offset+self.opt.patch_size,w_offset:w_offset+self.opt.patch_size])
            self.input_patch_list.append(self.real_A[:,:,h_offset:h_offset+self.opt.patch_size,w_offset:w_offset+self.opt.patch_size])

            # At this stage, we now have all the patch stuff in place, they will then be passed through the discriminator at a later stage
           # Honestly dont think the patch stuff is really being handled in the best way right now...

        # We are definitely switching the label here because the patches are fake but we're setting the 'target_is_real' variable to True. Look into this in accordance with MLM plus others
        accum_gen_loss=0
        pred_fake_patch_list= 0
        #Perform the predictions on the list of patches
        for i in range(self.opt.patchD_3):
            w_offset=random.randint(0,max(0,w-self.opt.patch_size-1))
            h_offset=random.randint(0,max(0,h-self.opt.patch_size-1))

            pred_fake_patch_list=self.L_Disc.forward(self.fake_patch_list[i])# Just check what is the shape of pred_fake_patch_list? Is it a single image or a batch?

            accum_gen_loss+=self.model_loss(pred_fake_patch_list,True)
        #Check if the below statement is really necessary( the dividing part)
        self.Gen_adv_loss+= accum_gen_loss/float(self.opt.patchD_3)
        # This now contains the loss from propagating the entire image and now, we're adding the average loss from all the fake patchs

        # Check if we use the other patch lists
        self.total_vgg_loss=self.vgg_loss.compute_vgg_loss(self.vgg,self.fake_B,self.real_A)*1.0 # This the vgg loss for the entire images!

        patch_loss_vgg=0
        for i in range(self.opt.patchD_3):
            patch_loss_vgg+=self.vgg_loss.compute_vgg_loss(self.vgg,self.fake_patch_list[i],self.input_patch_list[i])*1.0

        self.total_vgg_loss+=patch_loss_vgg/float(self.opt.patchD_3)

        self.Gen_loss=self.Gen_adv_loss+self.total_vgg_loss # The loss stuff is extremely simple to understand!
        self.Gen_loss.backward()# Compute the gradients of the generator using the sum of the adv loss and the vgg loss.



    def backward_D_basic(self,network,real,fake,is_global):
    # THIS IS ACTUALLY WHERE WE'RE WE TRAINING THE DISC SEPERATELY!
        pred_real=network.forward(real)
        pred_fake=network.forward(fake.detach())#< What does this even mean? I think that it may have something to do with how the gradients are calculated (but we shouldnt be caluclating gradients in the first place?)

        # Like in the generator case, this calculation is swapped for some reason. THIS IS THE FOUNDATION OF THE ENTIRE ALGORITHM. LOOK CAREFULLY INTO THIS EXPRESSION
        if(is_global): # Ragan is the relivistic discriminator! This label switching coincides with the paper. This is only used by the global discriminator
            Disc_loss=(self.model_loss(pred_real - torch.mean(pred_fake), True) +
            self.model_loss(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real=self.model_loss(pred_real,True)
            loss_D_fake=self.model_loss(pred_fake,False)
            Disc_loss=(loss_D_real+loss_D_fake)*0.5
        return Disc_loss

    def backward_G_Disc(self):
        self.G_Disc_loss=self.backward_D_basic(self.G_Disc,self.real_B,self.fake_B,True)
        self.G_Disc_loss.backward()

    def backward_L_Disc(self):
        L_Disc_loss= 0

        for i in range(self.opt.patchD_3):
            L_Disc_loss+=self.backward_D_basic(self.L_Disc, self.real_patch_list[i], self.fake_patch_list[i], False)
        # This is the normal calculation. The calc. for the whole image is handled seperately... we only handling patches here, thats why we can average everything.
        self.L_Disc_loss=L_Disc_loss/float(self.opt.patchD_3)
        self.L_Disc_loss.backward()


    def get_model_errors(self,epoch):
        Gen=self.Gen_loss.item()
        Global_disc=self.G_Disc_loss.item()
        Local_disc=self.L_Disc_loss.item()
        vgg=self.total_vgg_loss.item()/1.0
        return OrderedDict([('Gen',Gen),('G_Disc',Global_disc),('L_Disc',Local_disc),('vgg',vgg)])

    #Perfect
    def for_displaying_images(self):# Since self.realA_ was declared as a Variable, .data extracts the tensor of the variable.
        real_A=TensorToImage(self.real_A.data)# The low-light image (which is also our input image)
        fake_B=TensorToImage(self.fake_B.data)# Our produced result
        # What does the .data do? .data refrains from computing the gradient
        self_attention= TensorToImage(self.real_A_gray.data)
        return OrderedDict([('real_A', real_A),('fake_B', fake_B)])#, , ('latent_real_A', latent_real_A),('latent_show', latent_show), ('real_patch', real_patch),('fake_patch', fake_patch),('self_attention', self_attention)])

    def save_network(self,network,label,epoch):
        save_name='%s_net_%s.pth' %(epoch,label)
        save_path=os.path.join(self.opt.save_dir,save_name)
        torch.save(network.cpu().state_dict(),save_path)
        network.cuda(device=self.opt.gpu_ids[0])


    def save_model(self,label):
        self.save_network(self.Gen,'Gener',label)
        self.save_network(self.G_Disc,'Global_Disc',label)
        self.save_network(self.L_Disc,'Local_Disc',label)

    def load_model(self,network,network_name):
        list_of_files = glob.glob(str(self.opt.save_dir)+"/*") # * means all if need specific format then *.csv
        res = list(filter(lambda x: network_name in x, list_of_files))
        latest_file = max(res, key=os.path.getctime)
        loaded_file_path= os.path.join(self.opt.save_dir,latest_file)
        network.load_state_dict(torch.load(loaded_file_path))


def make_G(opt):
    generator=UnetGenerator(opt)
    generator.cuda(device=opt.gpu_ids[0])# jackpot! We see that the model is loaded to the GPU
    generator = torch.nn.DataParallel(generator, opt.gpu_ids)# We only need this when we have more than one GPU
    generator.apply(weights_init)# The weight initialization
    return generator


def make_Disc(opt,patch):
    discriminator=PatchGAN(opt,patch)
    discriminator.cuda(device=opt.gpu_ids[0]) # Jackpot, we are loading the model to the GPU
    discriminator = torch.nn.DataParallel(discriminator, opt.gpu_ids)# Split the input across all the GPU's (if applicable)
    discriminator.apply(weights_init)
    return discriminator


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss,self).__init__()
        self.Tensor=torch.cuda.FloatTensor
        self.loss=nn.MSELoss()

    def __call__(self,input,target_is_real):
        target_tensor= self.Tensor(input.size()).detach().fill_(float(target_is_real))
        return self.loss(input,target_tensor) # We then perform MSE on this!


def get_norm_layer(norm_type='instance'): # Optimize the position and the function itself.. Right now, it was directly copied over... Look into this
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    return norm_layer

class MinimalUnet(nn.Module):
    def __init__(self,down=None,up=None,submodule=None,withoutskip=False,**kwargs):
        super(MinimalUnet,self).__init__()

        self.down=nn.Sequential(*down) #THE LHS WILL HOLD THE RESULT!!!
        self.sub =submodule
        self.up= nn.Sequential(*up)
        self.withoutskip= withoutskip
        self.is_sub = not submodule == None # Will be false only for the innermost

    def forward(self,x,mask=None):
        if self.is_sub: # Almost recursive in a way
            x_up,_ = self.sub(self.down(x),mask)
        else: # If it is the inner-most (this would be the base case of the recursion)
            x_up = self.down(x)

        result= self.up(x_up)

        if self.withoutskip: # No skip connections are used for the outer layer
            x_out = result
        else:
            x_out = (torch.cat([x,result],1),mask)

        return x_out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module): # Perfect
    def __init__(self,opt):
        ngf=64
        super(UnetGenerator, self).__init__()

        norm_type=get_norm_layer(opt.norm_type)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,submodule=None,position='innermost', norm_layer=norm_type)
        for i in range(opt.num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8,submodule=unet_block, norm_layer=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8,submodule=unet_block, norm_layer=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,submodule=unet_block, norm_layer=norm_type)
        unet_block = UnetSkipConnectionBlock(3, ngf, submodule=unet_block,position='outermost', norm_layer=norm_type)# This is the outermost
        self.model=unet_block

    def forward(self, input):

        input, pad_left, pad_right, pad_top, pad_bottom = add_padding(input)
        latent= self.model(input[:,0:3,:,:],input[:,3:4,:,:])# Extraction is correct!
        latent = remove_padding(latent, pad_left, pad_right, pad_top, pad_bottom)
        input = remove_padding(input, pad_left, pad_right, pad_top, pad_bottom)
        return input[:,0:3,:,:]+latent


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, position='intermediate', norm_layer=nn.BatchNorm2d): # Has the attention stuff as well... look into it before adding it
        super(UnetSkipConnectionBlock, self).__init__()
        input_nc=outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,stride=2, padding=1) # The 3 is from looking at the encoder decoder approach
                             # Look into this!
        # Note that we we are not doing the double downsampling convolution
        downrelu = nn.LeakyReLU(0.2, True) # Look into the choice of activation function used!
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2,True)
        upnorm = norm_layer(outer_nc)

        if position=='outermost':
            up_conv= nn.ConvTranspose2d(2*inner_nc, outer_nc,kernel_size=4, stride=2, padding=1)
            #upsample=nn.Upsample(scale_factor = 2, mode='bilinear')
            #reflect = nn.ReflectionPad2d(1)
            #up_conv =nn.Conv2d(2*inner_nc,outer_nc,kernel_size=3, stride=1, padding=0)
            down = [downconv]
            #up= [uprelu,upsample,reflect,up_conv,nn.Tanh()]
            up = [uprelu, up_conv,nn.Tanh()]
            model = MinimalUnet(down,up,submodule,withoutskip=True)
        elif position=='innermost':
            upsample=nn.UpsamplingBilinear2d(scale_factor=2)
            down = [downrelu, downconv]
            up= [uprelu,upsample,upnorm]
            model = MinimalUnet(down,up)
        else:
            upsample=nn.UpsamplingBilinear2d(scale_factor=2)
            reflect = nn.ReflectionPad2d(1)
            up_conv =nn.Conv2d(2*inner_nc,outer_nc,kernel_size=3, stride=1, padding=0)
            #up_conv= nn.ConvTranspose2d(2*inner_nc, outer_nc,kernel_size=4, stride=2, padding=1,bias=use_bias)
            up= [uprelu,upsample,reflect,up_conv,upnorm]
            down = [downrelu, downconv,downnorm]
            #up = [uprelu,up_conv,upnorm]

            model = MinimalUnet(down,up,submodule)

        self.model =model

    def forward(self,x,mask=None):
        return self.model(x,mask)


class PatchGAN(nn.Module):
    def __init__(self,opt,patch):
        super(PatchGAN, self).__init__()

        self.opt=opt
        if patch:
            no_layers=self.opt.n_layers_patchD
        else:
            no_layers=self.opt.n_layers_D

        ndf=64
        # Needs to be treated seperately (as advised by Radford - we dont apply on output of generator and input of discriminator)
        sequence=[nn.Conv2d(3,ndf,kernel_size=4,stride=2,padding=2),
        nn.LeakyReLU(0.2,True)]

        nf_mult=1
        nf_mult_prev=1
        for n in range(1,no_layers):
            nf_mult_prev=nf_mult
            nf_mult=min(2**n,8)
            sequence+=[nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=4,stride=2,padding=2),
            nn.BatchNorm2d(ndf*nf_mult),
            nn.LeakyReLU(0.2,True)]

        nf_mult_prev=nf_mult
        nf_mult= min(2**no_layers,8)
        sequence+=[nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=4,stride=1,padding=2),
        nn.BatchNorm2d(ndf*nf_mult),
        nn.LeakyReLU(0.2,True)]

        sequence+=[nn.Conv2d(ndf*nf_mult,1,kernel_size=4,stride=1,padding=2)]


        self.model=nn.Sequential(*sequence)

    def forward(self,input):
        return self.model(input)#<-- pass through the discriminator itself which is represented by self.model


class PerceptualLoss(nn.Module):# All NN's needed to be based on this class and have a forward() function
    def __init__(self):
        super(PerceptualLoss,self).__init__()
        self.instance_norm=nn.InstanceNorm2d(512,affine=False)# <-- Affine determines if some of the parameters for normalizing are "learnable" or not.
        #512 is the number of features
		#This is to stabilize training

    def compute_vgg_loss(self,vgg_network,image,target):
        image_vgg=vgg_preprocess(image)
        target_vgg=vgg_preprocess(target)
        # The is precisely where we are calling forward on the vgg network
        img_feature_map=vgg_network(image_vgg)# Get the feature map of the input image
        target_feature_map=vgg_network(target_vgg)# Get the feature of the target image

        return torch.mean((self.instance_norm(img_feature_map) - self.instance_norm(target_feature_map)) ** 2)# The actual Perceptual Loss calculation


class Vgg(nn.Module):

    def __init__(self):
        super(Vgg,self).__init__()


        #This needs to be changed! Atleast the names... We were told ( check where) that the first 5 layers of the VGG networks are used.
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

    def forward(self,input):

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


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    return batch

def load_vgg(gpu_ids):
    vgg = Vgg()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load('vgg16.weight'))# Adding the weights to the model
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg
