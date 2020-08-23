import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from ManageData import TensorToImage, LatentToImage, AttentionToImage
import os
from collections import OrderedDict

#SPICE EVERYTHING UP, TOO SIMILIAR!!!
#Blend it with Mask Shadow GAN
# Find other papers as well
# Do with my own understanding!!!
#This is just temporary, be ruthless thereafter
# Check what would be the story when testing? Would a networks be created( sound silly but isnt!)

# Check the story about the transformation needed for preprocessing ( If RGB-> BGR is really necessary)


def weights_init(model):
    class_name=model.__class__.__name__
    if class_name.find('Conv')!=-1:
        torch.nn.init.normal_(model.weight.data,0.0,0.02)
    elif class_name.find('BatchNorm2d')!=-1:
        torch.nn.init.normal_(model.weight.data,1.0,0.02)
        torch.nn.init.constant_(model.bias.data,0.0)

class The_Model: # This is the grand model that encompasses everything ( the generator, both discriminators and the VGG network)
    def __init__(self,opt):

        self.opt=opt
        #I'm assuming that a CUDA GPU is used.
        self.input_A=torch.cuda.FloatTensor(opt.batch_size,3,opt.crop_size,opt.crop_size)#We are basically creating a tensor to store 16 low-light colour images with size crop_size x crop_size
        self.input_B=torch.cuda.FloatTensor(opt.batch_size,3,opt.crop_size,opt.crop_size)# Same as above but now for storing the normal-light images (NOT THE RESULT!)
        self.input_img=torch.cuda.FloatTensor(opt.batch_size,3,opt.crop_size,opt.crop_size)
        self.input_A_gray=torch.cuda.FloatTensor(opt.batch_size,1,opt.crop_size,opt.crop_size)# this is for the attention maps

        self.vgg_loss=PerceptualLoss(opt) # Use the alternate implementation when experimenting
        self.vgg_loss.cuda()#--> Shift to the GPU

        self.vgg=load_vgg(self.opt.gpu_ids)#This is for data parallelism
        self.vgg.eval() # We call eval() when some layers within the self.vgg network behave differently during training and testing... This will not be trained (Its frozen!)!
        #The eval function is often used as a pair with the requires.grad or torch.no grad functions (which makes sense)
        #I'm setting it to eval() because it's not being trained in anyway

        for weights in self.vgg.parameters():# THIS IS THE BEST WHY OF DOING THIS
            weights.requires_grad = False# Verified! For all the weights in the VGG network, we do not want to be updating those weights, therefore, we save computation using the above!


        self.Gen=make_G(opt)

        if(self.opt.isTrain): # Why would we be instantiating new discriminators when we are testing?? We shouldnt be coming here in the first place.
            self.G_Disc=make_Disc(opt,False) # This declaration should have a patch option because they have different number of layers each.
            self.L_Disc=make_Disc(opt,True) # Check if this switch is working correctly or not!

            self.model_loss=GANLoss() # They accept if we're using LSGAN and the type of tensore we're using ( These are standardized things in my implementation)

            #Check if the below optimizers are set in accordance with Radford!
            self.G_optimizer=torch.optim.Adam(self.Gen.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
            self.G_Disc_optimizer=torch.optim.Adam(self.G_Disc.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
            self.L_Disc_optimizer=torch.optim.Adam(self.L_Disc.parameters(),lr=opt.lr,betas=(opt.beta1,0.999))
            if self.opt.isTrain==False: # We shouldn't be coming here in the first place! We should directly be able to load the model...
                self.Gen.eval()# Do we really need this? I dont think that we are instantiating a new network when predicting, we're just loading an existing network...



    def perform_update(self,input):  #Do the forward,backprop and update the weights... this is a very powerful and 'highly abstracted' function
        # forward
    #This was directly copied over because the the stuff towards the bottom seemed necessary
        input_A=input['A']
        input_B=input['B']
        input_A_gray=input['A_gray']
        input_img=input['input_img']

        # This is extremely important and confusing!
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_img.resize_(input_img.size()).copy_(input_img)


        # Whatever comes below needs to be taken care of with extreme caution... Think everything through

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



    def forward(self):
        # Look into what the Variable stuff is for

        self.real_A=Variable(self.input_A)#Variable is basically a tensor (which represents a node in the comp. graph) and is part of the autograd package to easily compute gradients
        self.real_B=Variable(self.input_B) #This contains the normal-light images ( sort of our reference images)
        self.real_A_gray=Variable(self.input_A_gray) # This is the attention map
        self.real_img=Variable(self.input_img) #In our configuation, input_img=input_A


        #Make a prediction!
        # What is the latent used for?
        self.fake_B,self.latent_real_A= self.Gen.forward(self.real_img,self.real_A_gray)# We forward prop. a batch at a time, not individual images in the batch!

        # Experiment as much as possible with the latent variable and understand what exactly does it represent. Find a better way of doing the cropping, their approach looks lame...
        w=self.real_A.size(3)
        h=self.real_B.size(2)
        # Remove this individual patch!
        w_offset=random.randint(0,max(0,w-self.opt.patch_size-1))
        h_offset=random.randint(0,max(0,h-self.opt.patch_size-1))

        # Check if there is really a need for these seperate patches
        # fake_B is a tensor of many images, how do we know from which image in the tensor are we cropping from? It seems that we take a patch from each image in the tensor (containing 16 images each)
        self.fake_patch=self.fake_B[:,:,h_offset:h_offset+self.opt.patch_size,w_offset:w_offset+self.opt.patch_size] # Patch from our enhanced result
        self.real_patch=self.real_B[:,:,h_offset:h_offset+self.opt.patch_size,w_offset:w_offset+self.opt.patch_size]# Path from the training set's normal light images
        self.input_patch=self.real_A[:,:,h_offset:h_offset+self.opt.patch_size,w_offset:w_offset+self.opt.patch_size]# patch from the training set's low-light images... I dont really see why do we need this though??
        #print(self.fake_patch.size()) [16,3,32,32]

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

    def backward_G(self):
        # First let the discriminator make a prediction on the fake samples
        #This is the part recommended by Radford where we test real and fake samples in stages
        pred_fake=self.G_Disc.forward(self.fake_B)
        # torch.mean() is a scalar... what is going on?
        pred_real=self.G_Disc.forward(self.real_B)

        # We are switching the labels when calculating the adversarial loss, I dont understand the subtraction of the opposite mean
        #CONFIRM THE SWITCHING STORY!!! What the hell is going on here? Why are the TERMS switched??? JUST THE ENTIRE FORM OF THE BELOW EXPRESSION LOOKS EXTREMELY FISHY!!! Note that we are taking the average ( dividing by 2 for some reason... Dont reason think this is necessary?!) Link this function to the __call__ function of GAN Loss
        # INNOVATE HERE!
        self.Gen_adv_loss= (self.model_loss(pred_real  - torch.mean(pred_fake), False) + self.model_loss(pred_fake  - torch.mean(pred_real), True)) / 2
        # In a seperate variable, we start accumulating the loss from the different aspects (which include the loss on the patches and the vgg loss)

        accum_gen_loss=0
        #Still dont see the point of this...
        pred_fake_patch=self.L_Disc.forward(self.fake_patch)
        # We are definitely switching the label here because the patches are fake but we're setting the 'target_is_real' variable to True. Look into this in accordance with MLM plus others

        accum_gen_loss+=self.model_loss(pred_fake_patch,True)# Here, we are definitely swapping the label

        #Perform the predictions on the list of patches
        for i in range(self.opt.patchD_3):
            pred_fake_patch_list=self.L_Disc.forward(self.fake_patch_list[i])# Just check what is the shape of pred_fake_patch_list? Is it a single image or a batch?

            accum_gen_loss+=self.model_loss(pred_fake_patch_list,True)
        #Check if the below statement is really necessary( the dividing part)
        self.Gen_adv_loss+= accum_gen_loss/float(self.opt.patchD_3+1)
        # This now contains the loss from propagting the entire image and now, we're adding the average loss from all the fake patchs

        # Check if we use the other patch lists
        self.total_vgg_loss=self.vgg_loss.compute_vgg_loss(self.vgg,self.fake_B,self.real_A)*1.0 # This the vgg loss for the entire images!

        #patch_loss_vgg=0
        patch_loss_vgg=self.vgg_loss.compute_vgg_loss(self.vgg,self.fake_patch,self.input_patch)*1.0
        # This is the vgg loss for the individual patch
        # Check what its the diff between self.input_patch and self.real_patch? I think input_patch is the low-light patches and real_patches are the normal_light images,thats why I'd be wrong.
        for i in range(self.opt.patchD_3):
            patch_loss_vgg+=self.vgg_loss.compute_vgg_loss(self.vgg,self.fake_patch_list[i],self.input_patch_list[i])*1.0

        self.total_vgg_loss+=patch_loss_vgg/float(self.opt.patchD_3+1)

        self.Gen_loss=self.Gen_adv_loss+self.total_vgg_loss # The loss stuff is extremely simple to understand!
        self.Gen_loss.backward()# Compute the gradients of the generator using the sum of the adv loss and the vgg loss.



    def backward_D_basic(self,network,real,fake,use_ragan):
    # THIS IS ACTUALLY WHERE WE'RE WE TRAINING THE DISC SEPERATELY!
        pred_real=network.forward(real)
        pred_fake=network.forward(fake.detach())#< What does this even mean? I think that it may have something to do with how the gradients are calculated (but we shouldnt be caluclating gradients in the first place?)

        # Like in the generator case, this calculation is swapped for some reason. THIS IS THE FOUNDATION OF THE ENTIRE ALGORITHM. LOOK CAREFULLY INTO THIS EXPRESSION
        if(use_ragan):
            Disc_loss=(self.model_loss(pred_real - torch.mean(pred_fake), True) +
            self.model_loss(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real=self.model_loss(pred_real,True)
            loss_D_fake=self.model_loss(pred_fake,False)
            Disc_loss=(loss_D_real+loss_D_fake)*0.5
        return Disc_loss

    def backward_G_Disc(self):
        # Try to think carefully about why are we're in doing the following... We're training the discriminator using the 'real' (normal light images) and the fake samples... Why are we doing this again? We just did something similiar when updating the generator
        # It's probably just for the sequencing>>> We have to update the generator before turning our attention to the discriminator
        self.G_Disc_loss=self.backward_D_basic(self.G_Disc,self.real_B,self.fake_B,True)
        self.G_Disc_loss.backward()# Thing about where exactly are we backpropagating this!?

    def backward_L_Disc(self):
        L_Disc_loss=self.backward_D_basic(self.L_Disc,self.real_patch,self.fake_patch,False)

        for i in range(self.opt.patchD_3):
            L_Disc_loss+=self.backward_D_basic(self.L_Disc, self.real_patch_list[i], self.fake_patch_list[i], False)
        # They is the normal calculation. The calc. for the whole image is handled seperately... we only handling patches here, thats why we can average everything.
        self.L_Disc_loss=L_Disc_loss/float(self.opt.patchD_3+1)
        self.L_Disc_loss.backward()


    def get_model_errors(self,epoch):
        Gen=self.Gen_loss.item()
        Global_disc=self.G_Disc_loss.item()
        Local_disc=self.L_Disc_loss.item()
        vgg=self.total_vgg_loss.item()/1.0
        return OrderedDict([('Gen',Gen),('G_Disc',Global_disc),('L_Disc',Local_disc),('vgg',vgg)])

    def for_displaying_images(self):# Since self.realA_ was declared as a Variable, .data extracts the tensor of the variable.
        real_A=TensorToImage(self.real_A.data)# The low-light image (which is also our input image)
        fake_B=TensorToImage(self.fake_B.data)# Our produced result

        # Experiment alot to see what is this latent stuff and what is it used for
        # What does the .data do?
        latent_real_A=TensorToImage(self.latent_real_A.data)
        latent_show=LatentToImage(self.latent_real_A.data)


        self_attention= AttentionToImage(self.real_A_gray.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])#, ('latent_real_A', latent_real_A),('latent_show', latent_show), ('real_patch', real_patch),('fake_patch', fake_patch),('self_attention', self_attention)])

    def save_network(self,network,label,epoch):
        save_name='%s_net_%s.pth' %(epoch,label)
        save_path=os.path.join(self.opt.save_dir,save_name)
        torch.save(network.cpu().state_dict(),save_path)
        network.cuda(device=self.opt.gpu_ids[0])


    def save_model(self,label):
        self.save_network(self.Gen,'Gener',label)
        self.save_network(self.G_Disc,'Global_Disc',label)
        self.save_network(self.L_Disc,'Local_Disc',label)



def make_G(opt):
    generator=Unet_generator1(opt)
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
        self.real_label=1.0
        self.fake_label=0.0
        #Check the need for the var stuff?
        self.real_label_var=None
        self.fake_label_var=None
        self.Tensor=torch.cuda.FloatTensor
        self.loss=nn.MSELoss()

    # #This function is absolute rubbish! Can be significantly improved!
    # def get_target_tensor(self,input,target_is_real):#This function basically creates a target label tensor which is used to compute the MSE in __call__.
    #     target_tensor=None
    #     if target_is_real:
    #         target_tensor=Variable(self.Tensor(input.size()).fill_(self.real_label),requires_grad=False)
    #     else:
    #         target_tensor=Variable(self.Tensor(input.size()).fill_(self.fake_label),requires_grad=False)
    #     return target_tensor


    #This function is absolute rubbish! Can be significantly improved!
    def get_target_tensor(self,input,target_is_real):#This function basically creates a target label tensor which is used to compute the MSE in __call__.
        target_tensor=None
        if target_is_real:
            #This is a boolean for whether we need to actually create a new label tensor
            create_label=((self.real_label_var is None) or (self.real_label_var.numel()!=input.numel()))

            if create_label:
                real_tensor=self.Tensor(input.size()).fill_(self.real_label)
                #Check why do we need the variable function
                self.real_label_var=Variable(real_tensor,requires_grad=False)
            target_tensor=self.real_label_var
        else:
            create_label=((self.fake_label_var is None) or (self.fake_label_var.numel()!=input.numel()))
            if create_label:
                fake_tensor=self.Tensor(input.size()).fill_(self.fake_label)
                #Check why do we need the variable function
                self.fake_label_var=Variable(fake_tensor,requires_grad=False)
            target_tensor=self.fake_label_var
        return target_tensor

    def __call__(self,input,target_is_real):
        target_tensor=self.get_target_tensor(input,target_is_real)
        return self.loss(input,target_tensor) # We then perform MSE on this!


class Unet_generator1(nn.Module):
    def __init__(self,opt):
        super(Unet_generator1,self).__init__()

        self.opt=opt
        # Surely I only need one of the MaxPooling layers, they are pretty much identical!
        # These will be used to resize the attention map to fit the latent result at each upsampling step
        # Try to avoid MaxPool and duplicacy
        self.att_downsize = nn.MaxPool2d(2)# This is seperate (this is for the attention maps to fit the size of the filters in each layer) -----> This is for downsampling the attention map. At each step, the size of the attention map is halved.

        # I must just create a function to create these module. Ridiculously shoddy!

        self.conv1_1=nn.Conv2d(4,32,3,padding=1)# 4 because of the the RGB image and the attention map...
        self.LRelu1_1=nn.LeakyReLU(0.2,inplace=True) # Inplace is to make the changes directly without producing additional output (it is what it says it is)
        if(self.opt.norm_type=='Batch'): # I have batch by they have it as use_norm which is the L1 loss weight
            self.norm1_1=nn.BatchNorm2d(32)
        else:
            self.norm1_1=nn.InstanceNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.LRelu1_2 = nn.LeakyReLU(0.2, inplace=True)
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
        self.LRelu2_2 = nn.LeakyReLU(0.2, inplace=True)
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
        self.LRelu3_2 = nn.LeakyReLU(0.2, inplace=True)
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
        self.LRelu4_2 = nn.LeakyReLU(0.2, inplace=True)
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
        self.LRelu5_2 = nn.LeakyReLU(0.2, inplace=True)
        if (opt.norm_type=='Batch'):
            self.norm5_2 = nn.BatchNorm2d(512)
        else:
            self.norm5_2=nn.InstanceNorm2d(512)
        self.max_pool5 = nn.MaxPool2d(2)# OVer and above everything, these maxpools can be removed because each is not different from each other.


        #The bottleneck has been reached, we now enter the decoder. We need to now upsample to produce the sample.
        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)# IT SEEMS THAT THEY ALREADY ATTEMPTED WHAT I WANTED TO DO( USE THE TRANSPOSE CONV LAYER TO UPSAMPLE)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=1) # Try to get an intuition of how the no. of filters,kernel_size and strides are configured to achieve different characteristics
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1) # In the forward pass, this will be used with the bilinear parameter
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
        # Look into this tanh function to ensure that we are withing [-1,1]
        #self.tanh= nn.Tanh()# In the provided training conf., tanh is not used. But how do we ensure that the output is within an acceptable range?

    def forward(self,input,gray): # We forward propagate a batch at a time!
        # flag=0
        # if input.size()[3] >2200:# This seems ridiculous! Test performance when this is removed
        #     avg=nn.avgPool2d(2)
        #     input=avg(input)
        #     gray=avg(gray)
        #     flag=1#--> Indicates that at the end, we need to upsample
            # Before Performing a forward pass on the tensor, we first pad the tensor containing the real (low-light) images
			#If the dimensions of the images are perfectly divisible by 16, we dont pad.
			# Otherwise, we pad the dimensions that are skew by the amount such that the dim. of the new padded version is divisible by 16.
			#The pad_tensor function performs the padding (if necessary) and returns how much padding was applied to each side which makes it easier when removing the padding later.


        # First downsample the attention map for all stages
        gray_2=self.att_downsize(gray)
        gray_3=self.att_downsize(gray_2)
        gray_4=self.att_downsize(gray_3)
        gray_5=self.att_downsize(gray_4)


        #Gray_2 size: torch.Size([16, 1, 160, 160])
        #Gray_3 size: torch.Size([16, 1, 80, 80])
        #Gray_4 size: torch.Size([16, 1, 40, 40])
        #Gray_5 size: torch.Size([16, 1, 20, 20])

        #Input Size: torch.Size([16, 3, 320, 320])

        #Surely below can be automated!!!, do right at the end when I know what I'm doing!
        x=self.norm1_1(self.LRelu1_1(self.conv1_1(torch.cat((input,gray),1))))

        conv1=self.norm1_2(self.LRelu1_2(self.conv1_2(x)))
        x=self.max_pool1(conv1)

        x=self.norm2_1(self.LRelu2_1(self.conv2_1(x)))
        conv2=self.norm2_2(self.LRelu2_2(self.conv2_2(x)))
        x=self.max_pool2(conv2)

        x=self.norm3_1(self.LRelu3_1(self.conv3_1(x)))
        conv3=self.norm3_2(self.LRelu3_2(self.conv3_2(x)))
        x=self.max_pool3(conv3)

        x=self.norm4_1(self.LRelu4_1(self.conv4_1(x)))
        conv4=self.norm4_2(self.LRelu4_2(self.conv4_2(x)))
        x=self.max_pool4(conv4)

        x=self.norm5_1(self.LRelu5_1(self.conv5_1(x)))
        x=x*gray_5
        conv5=self.norm5_2(self.LRelu5_2(self.conv5_2(x)))

        #Bottleneck has been reached( I think, but then, why is the att map already being multiplied?) - start upsampling
        # Experiment here to see if bilinear upsampling really is this best option.

        conv5=F.upsample(conv5,scale_factor=2,mode='bilinear')
        conv4=conv4*gray_4
        up6=torch.cat([self.deconv5(conv5),conv4],1)
        x=self.norm6_1(self.LRelu6_1(self.conv6_1(up6)))
        conv6=self.norm6_2(self.LRelu6_2(self.conv6_2(x)))

        conv6=F.upsample(conv6,scale_factor=2,mode='bilinear')
        conv3=conv3*gray_3
        up7=torch.cat([self.deconv6(conv6),conv3],1)
        x=self.norm7_1(self.LRelu7_1(self.conv7_1(up7)))
        conv7=self.norm7_2(self.LRelu7_2(self.conv7_2(x)))

        conv7=F.upsample(conv7,scale_factor=2,mode='bilinear')
        conv2=conv2*gray_2
        up8=torch.cat([self.deconv7(conv7),conv2],1)
        x=self.norm8_1(self.LRelu8_1(self.conv8_1(up8)))
        conv8=self.norm8_2(self.LRelu8_2(self.conv8_2(x)))

        conv8=F.upsample(conv8,scale_factor=2,mode='bilinear')
        conv1=conv1*gray
        up9=torch.cat([self.deconv8(conv8),conv1],1)
        x=self.norm9_1(self.LRelu9_1(self.conv9_1(up9)))
        conv9=self.LRelu9_2(self.conv9_2(x))

        latent = self.conv10(conv9)# What is this for?
        #Latent Size = torch.Size([16, 3, 320, 320])

        latent = latent*gray

        #if self.opt.tanh:
        #    latent = self.tanh(latent)# Oddly does not apply to us
        output = latent + input*float(self.opt.skip)# This is a breakthrough! The latent result is added to the low-light image to form the output.

        # if flag == 1: # If fineSize>2200 which resulting in having to perform AvgPooling
        #     output = F.upsample(output, scale_factor=2, mode='bilinear')
        #     gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        return output, latent # Want to see what is this latent!


class PatchGAN(nn.Module): # Make sure the configuration of the PatchGAN is absolutely "textbook stuff"
    def __init__(self,opt,patch):
        super(PatchGAN, self).__init__()

        self.opt=opt
        if patch:
            no_layers=self.opt.n_layers_patchD
        else:
            no_layers=self.opt.n_layers_D

        sequence=[nn.Conv2d(3,64,kernel_size=4,stride=2,padding=2),nn.LeakyReLU(0.2,True)]
        # The rubbish below can be modified!
        #Filter out this rubbish ( Warning 1 - its the simple input output procedure)
        # Output collapses from 512 - 1... Check what does the 1 activation map represent
        # Needs to be completely restructured according to Radford
        # Check if we need to include sigmoid at the end?
        # Look at flagship papers, as well as morphing EGAN's alternate implementations and get ideas from other papers
        ndf=64
        nf_mult=1
        nf_mult_prev=1
        for n in range(1,no_layers):
            nf_mult_prev=nf_mult
            nf_mult=min(2**n,8)
            sequence+=[nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=4,stride=2,padding=2),
            nn.LeakyReLU(0.2,True)]

        nf_mult_prev=nf_mult
        nf_mult= min(2*no_layers,8)
        sequence+=[nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,kernel_size=4,stride=1,padding=2),nn.LeakyReLU(0.2,True)]

        sequence+=[nn.Conv2d(ndf*nf_mult,1,kernel_size=4,stride=1,padding=2)]

        # Read up on the story about the sigmoid at the end. If yes, += it here

        self.model=nn.Sequential(*sequence)

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
        #print("I am in Compute VGG_loss")
        #print(image.shape)
        image_vgg=vgg_preprocess(image)#cv2.normalize(image,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#vgg_preprocess(image,self.opt)--> This function was supposed to convert the RGB image to BGR and convert the normalized image [-1,1] from the tanh function to [0,255]... Im removing it now, but check ifit is really necessary to change the range.
        target_vgg=vgg_preprocess(target)#cv2.normalize(target,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#vgg_preprocess(target,self.opt)
        # Check if there is a work around this!

        img_feature_map=vgg_network(image_vgg,self.opt)# Get the feature map of the input image
        target_feature_map=vgg_network(target_vgg,self.opt)# Get the feature of the target image

        return torch.mean((self.instance_norm(img_feature_map) - self.instance_norm(target_feature_map)) ** 2)# --> According to the provided function


class Vgg(nn.Module): # optimize this, There should surely be some variations to this.. Understand what is trying to be achieved and then determine how to go about achieving this!


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

    def forward(self,input,opt):

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


#There is a lot of room for variation here
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
