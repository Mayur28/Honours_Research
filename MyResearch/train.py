from SetupTraining import SetupTraining
from ManageData import DataLoader
import Networks
import time



opt=SetupTraining().parse()# This is obviously just a start, I can (and must) adjust it accordingly
#Just check if I really need config!
data_loader=DataLoader(opt)
dataset=data_loader.load()
print("Number of training Images: %d"% len(data_loader))


the_model=Networks.The_Model(opt)

total_steps=0
# Below is the big deal!!! range(1,100+100+1)# the lr decays for last 100 epochs
for epoch in range(1,opt.opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    for i, data in enumerate(dataset): # For each call, __get_item__ is called for each image in the current batch. Takes the images, formats it into the desired dictionary format, and this dictionary is then represented by data
        iter_start_time=time.time()
        total_steps+=op.batch_size
        epoch_iter=total_steps-len(data_loader)*(epoch-1)
        #Remember at this stage, data is the batch 'dataset' in dictionary format. It slots the data into the correct variables self.inputA,etc to easily perform propagation operations
        model.perform_update(epoch)
        
        