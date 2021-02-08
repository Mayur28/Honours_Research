import os
import torch
import argparse
import time
import numpy as np

# For inference, the the data_source should be set to ../test_dataset. For training and testing, use final_dataset

def DefaultSetup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='TheModel', help="Name of the current execution")
    parser.add_argument('--data_source', type=str, default='../test_dataset', help='path to dataset and should have substructure containing trainA,trainB,testA,testB')
    # Calibrate the batch batch_size, crop_size and patch_size
    parser.add_argument('--crop_size', type=int, default=512, help='This will be the the size of the input to our network (the size is reduced by RandomCropping)')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patch')
    parser.add_argument('--gpu_ids', type=str, default='0', help="Used to specify the id's of the GPU's (more specifically, '0' for 1 GPU, '0,1' for 2 GPU's,etc)")
    parser.add_argument('--checkpoints_dir', type=str, default='/content/drive/My Drive/Low-light_Image_Enh/', help='models are saved here')
    parser.add_argument('--norm_type', type=str, default='batch', help='instance or batch normalization in the generator')
    parser.add_argument('--num_downs', type=int, default=9, help=' How many U-net modules are created in the generator')
    parser.add_argument('--num_disc_layers', type=int, default=7, help='number of layers in global discriminator')
    parser.add_argument('--num_patch_disc_layers', type=int, default=6, help='number of layers in local discriminator')
    parser.add_argument('--num_patches', type=int, default=7, help='Number of patches to crop for the local discriminator')
    return parser


def TrainingSetup(the_args):
    the_args.add_argument('--batch_size', type=int, default=8, help='input batch size (One of the aspects that can be used to control GPU requirements)')
    the_args.add_argument('--phase', type=str, default='train', help='train or test')
    the_args.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    the_args.add_argument('--niter_decay', type=int, default=50, help='# of epochs to decay the learning rate')
    the_args.add_argument('--beta1', type=float, default=0.5, help='momentum term of Adam')
    the_args.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for Adam')
    the_args.add_argument('--display_freq', type=int, default=30, help='frequency of showing training results on screen')
    the_args.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    the_args.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    return the_args


def TestingSetup(the_args):
    the_args.add_argument('--batch_size', type=int, default=1, help='input batch size (One of the aspects that can be used to control GPU requirements)')
    the_args.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    return the_args


def process(the_args):
    opt = the_args.parse_args()

    opt.gpu_ids = list(map(int, opt.gpu_ids.split(',')))

    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    # below creates the necessary directories (for storing the results)
    args = vars(opt)
    if not os.path.isdir(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.isdir(expr_dir):
        os.mkdir(expr_dir)
    file_name = os.path.join(expr_dir, 'config.txt')

    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')  # First statement will print to the file
        print('------------ Options -------------')  # Second statement will print to terminal

        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))

        opt_file.write('-------------- End ----------------\n')
        print('-------------- End ----------------')
    if opt.phase == 'train':
        opt.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'Training_IO')
    else:
        opt.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'Testing_IO')
    print(opt.img_dir)
    if not os.path.isdir(opt.img_dir):
        os.mkdir(opt.img_dir)
    opt.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(opt.log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
    # Where models are stored
    opt.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    return opt
