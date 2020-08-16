import os
import torch
import argparse
import time

#(Change the 'help' eventually!)
# vgg_choose will be set to relu5_1. Remove the if-statements in networks.py--> Check why was this option used
# Examine their multiple approaches again and try to combine uniquely
# They are using maxpooling in the generator, not avg_pooling... Check Radford's appraoch to downsampling
# I want tanh at the end of mine!--> Check if this would break the definition of an LSGAN
# Default setting doesn't use 'lighten' which normalizes the attention map... Experiment with this! Only appears just before
#the attention map calculation...
# Theres actually a lot that I removed from 'UnalignedDataset' that appears to relate to data augmentation... Experiment with this
# What does pool_size do and affect results?

class SetupTraining():
    def __init__(self):
        self.parser=argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='TheModel', help="Name of the current execution")
        self.parser.add_argument('--display_freq', type=int, default=30, help='frequency of showing training results on screen')

        # Calibrate the batch batch_size, crop_size and patch_size
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size (One of the aspects that can be used to control GPU requirements)')
        self.parser.add_argument('--crop_size', type=int, default=320, help='Crop the images to this new size')
        self.parser.add_argument('--patch_size', type=int, default=32, help='specifies the size of the patch that we are going to use')
        # This can be modified according to the number of GPU's used to train the model
        self.parser.add_argument('--gpu_ids', type=str, default='0', help="Used to specify the ID's of the GPU's to be used (e.g 0,1 if 2 GPU's are used)")

        self.parser.add_argument('--checkpoints_dir', type=str, default='/content/drive/My Drive/MyResearch/', help='models are saved here')
        self.parser.add_argument('--norm_type', type=str, default='batch', help='instance normalization or batch normalization')
        # We are only cropping, experiment with the other options here!
        self.parser.add_argument('--resize_or_crop', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')


        self.parser.add_argument('--skip', default=True, help='B = net.forward(A) + skip*A')
        self.parser.add_argument('--use_mse', action='store_true', help='MSELoss')
        # use_norm will be set to true by default
        self.parser.add_argument('--use_ragan', default=True, action='store_true', help='use ragan')
        #Sort out the VGG stuff!
        self.parser.add_argument('--vgg', type=float, default=1.0, help='use perceptrual loss')
        # vgg_mean was false!
        # vgg_choose will be set to relu5_1. Remove the if-statements in networks.py
        # For vgg_choose, check what was the purpose of having so many different options.
        # Examine their multiple approaches again and train to combine uniquely
        #no_vgg_instance=False
        # vgg_maxpooling=False
        # use_avgpool specifies if we use average or max pooling... Experiment with this!
        self.parser.add_argument('--n_layers_D', type=int, default=5, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_patchD', type=int, default=4, help='only used if which_model_netD==n_layers')
        # Maxpooling is used instead of avg_pooling to downsample in the generator
        self.parser.add_argument('--use_avgpool', type=float, default=0, help='use perceptrual loss')
        # They included an option that adds instance_norm before vgg to stabilize training
        # noise will be set to 0 by default
        #input_linear is false
        #patchD will be True by default
        self.parser.add_argument('--patchD_3', type=int, default=5, help='choose the number of crop for patch discriminator')
        # We dont use D_P_times2, what why was it considered in the first place?
        self.parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
        self.parser.add_argument('--patch_vgg', default=True, action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--hybrid_loss', default=True, action='store_true', help='use lsgan and ragan separately')
        self.parser.add_argument('--self_attention', default= True,  action='store_true', help='adding attention on the input of generator')

        # We have this! What does it do? Multiplies the latent result to the attention map in the generator... But why?
        self.parser.add_argument('--times_residual', default=True, action='store_true', help='output = input + residual*attention')
        #Now, the proper training options
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        # EGAN used 0.0001 but Radford recommended 0.0002
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        # Configure the pool size, what exactly does it do?
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.isTrain = True


    def process(self): # This is sorted!
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        expr_dir = os.path.join(self.opt.checkpoints_dir, "TheModel")
        if(os.path.isdir(expr_dir)==False):
            os.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'config.txt')

        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')# First statement will print to the file
            print('------------ Options -------------')# Second statement will print to terminal

            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))

            opt_file.write('-------------- End ----------------\n')
            print('-------------- End ----------------')

        self.opt.web_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'web')
        self.opt.img_dir = os.path.join(self.opt.web_dir, 'images')
        print('create web directory %s...' % self.opt.web_dir)
        if(os.path.isdir(self.opt.web_dir)==False):
            os.mkdir(self.opt.web_dir)
        if(os.path.isdir(self.opt.img_dir)==False):
            os.mkdir(self.opt.img_dir)
        self.opt.log_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'loss_log.txt')
        with open(self.opt.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.opt.save_dir=os.path.join(opt.checkpoints_dir,opt.name)
        return self.opt
