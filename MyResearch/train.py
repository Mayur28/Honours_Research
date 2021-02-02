from Setup import *
from ManageData import DataLoader
from PIL import Image
import Networks
import time
import os


# Saves the input and output images
def save_images(images, title, phase='train'):
    for label, image in images.items():  # .items() extracts the "packages" from the dictionary
        img_path = os.path.join(opt.img_dir, 'epoch%.3d_%s.png' % (title, label))
        image_pil = Image.fromarray(image)
        image_pil.save(img_path)

# Prints the errors of the generator (+ vgg loss) and both discriminators, making it easier to detect model collapse
def print_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f)' % (epoch, i, t)
    for k, v in errors.items():  # --> This is to extract from the Ordered Dictionary
        message += '%s: %.3f ' % (k, v)
    print(message)
    with open(opt.log_name, "a") as log_file:
        log_file.write('%s\n' % message)


opt = process(TrainingSetup(DefaultSetup())) # Parse the training options that will be used
data_loader = DataLoader(opt)
dataset = data_loader.load() # Load the training dataloader
print("Number of training images: %d" % len(data_loader))
the_model = Networks.The_Model(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay+ 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):  # For each call, __get_item__ is called for each image in the current batch. Takes the images, formats it into the desired dictionary format, and this dictionary is then represented by data

        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - len(data_loader) * (epoch - 1)
        the_model.set_input(data) # Insert the new data into the necessary containers to be read from during the forward and backward pass
        the_model.perform_update()

        # Below prints diagnostic information such as time taken per an epoch
        if total_steps % opt.display_freq == 0:
            save_images(the_model.for_displaying_images(), epoch)

        if total_steps % opt.print_freq == 0:
            exec_time = (time.time() - iter_start_time) / opt.batch_size
            print_errors(epoch, epoch_iter, the_model.get_model_errors(epoch), exec_time)

    if epoch % opt.save_epoch_freq == 0:
        the_model.save_model(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter, time.time() - epoch_start_time))

    # Detects when do we start decaying the learning rate (apparently improves results so that "the model does not get trapped in a local minima")
    if(epoch> opt.niter):
        the_model.update_learning_rate()
