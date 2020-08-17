from SetupTraining import SetupTraining
from ManageData import DataLoader
from PIL import Image
import Networks
import time
import os


def display_current_results(images,epoch):
    for label,image in images.items():
        img_path=os.path.join(opt.img_dir,'epoch%.3d_%s.png'%(epoch,label))
        image_pil=Image.fromarray(image)
        image_pil.save(img_path)

def print_errors(epoch,i,errors,t):
    message='(epoch: %d, iters: %d, time: %.3f)'%(epoch,i,t)
    for k,v in errors.items():# --> This is to extract from the Ordered Dictionary
        message+= '%s: %.3f ' % (k, v)
    print(message)
    with open(opt.log_name, "a") as log_file:
        log_file.write('%s\n' % message)


opt=SetupTraining().process()
data_loader=DataLoader(opt)
dataset=data_loader.load()
print("Number of training Images: %d"% len(data_loader))
the_model=Networks.The_Model(opt)

total_steps=0

# Below is the big deal!!! range(1,100+100+1)# the lr decays for last 100 epochs
for epoch in range(1,opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    for i, data in enumerate(dataset): # For each call, __get_item__ is called for each image in the current batch. Takes the images, formats it into the desired dictionary format, and this dictionary is then represented by data

        iter_start_time=time.time()
        total_steps+=opt.batch_size
        epoch_iter=total_steps-len(data_loader)*(epoch-1)
        #Remember at this stage, data is the batch 'dataset' in dictionary format. It slots the data into the correct variables self.inputA,etc to easily perform propagation operations
        the_model.perform_update(data)

        if(total_steps% opt.display_freq==0):
            display_current_results(the_model.for_displaying_images(),epoch)

        if(total_steps% opt.print_freq==0):
            exec_time=(time.time()-iter_start_time)/opt.batch_size
            print_errors(epoch,epoch_iter,the_model.get_model_errors(epoch),exec_time)

    if(epoch% opt.save_epoch_freq==0):
        the_model.save_model(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    #if(epoch>opt.niter):
    #    model.update_learning_rate()
