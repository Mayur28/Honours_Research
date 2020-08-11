import os
import torch
import argparse

class SetupTraining():
    def __init__(self):
        self.parser=argparse.ArgumentParser()
        self.parser.add_argument('--name', type=str, default='TheModel', help="Name of the current execution")
        self.parser.add_argument('--display_freq', type=int, default=30, help='frequency of showing training results on screen')
        # These are the BaseOptions (Change the 'help' eventually!)
        #self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size (One of the aspects that can be used to control GPU requirements)')
        self.parser.add_argument('--crop_size', type=int, default=320, help='Crop the images to this new size')
        self.parser.add_argument('--patch_size', type=int, default=32, help='specifies the size of the patch that we are going to use')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help="Used to specify the ID's of the GPU's to be used (e.g 0,1 if 2 GPU's are used)")
        #Ours will alls be in accordance with the unaligned dataset
        # We are only using the single_model
        # Set the number of loading threads to 6 ( there's is 4)
        #self.parser.add_argument('--checkpoints_dir', type=str, default='/content/drive/My Drive/EnlightenGAN2/EnlightenGAN-master/checkpoints/', help='models are saved here')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/content/drive/My Drive/EnlightenGAN2/', help='models are saved here')
        self.parser.add_argument('--norm_type', type=str, default='batch', help='instance normalization or batch normalization')
        # We are only cropping, experiment with the other options here!
        self.parser.add_argument('--resize_or_crop', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')


        self.parser.add_argument('--skip', type=float, default=0.8, help='B = net.forward(A) + skip*A')
        self.parser.add_argument('--use_mse', action='store_true', help='MSELoss')
        # use_norm will be set to true by default
        self.parser.add_argument('--use_ragan', action='store_true', help='use ragan')
        #Sort out the VGG stuff!
        self.parser.add_argument('--vgg', type=float, default=1, help='use perceptrual loss')
 # vgg_mean was false!
# vgg_choose will be set to relu5_3. Remove the if-statements in networks.py
#no_vgg_instance=False
 # vgg_maxpooling=False
    # In vgg is false
    # use_avgpool specifies if we use average or max pooling... Experiment with this!
        self.parser.add_argument('--n_layers_D', type=int, default=5, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_patchD', type=int, default=4, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--use_avgpool', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--instance_norm', type=float, default=0, help='use instance normalization')
        # I want tanh at the end of mine!
        # noise will be set to 0 by default
        #input_linear is false
        #patchD will be True by default
        self.parser.add_argument('--patchD_3', type=int, default=5, help='choose the number of crop for patch discriminator')
        # We use D_P_times2, but look into it further. The vgg stuff can be considerably simplified.
        self.parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--hybrid_loss', action='store_true', help='use lsgan and ragan separately')
        self.parser.add_argument('--self_attention', action='store_true', help='adding attention on the input of generator')
        # We have this!
        self.parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')
        #Now, the proper training options
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--config', type=str, default='configs/unit_gta2city_folder.yaml', help='Path to the config file.')
        self.isTrain = True


    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, "TheModel")
        if(os.path.isdir(expr_dir)==False):
            os.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
