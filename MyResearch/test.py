from SetupTraining import *
from ManageData import DataLoader


opt=TestSetup(DefaultSetup())
opt.no_threads= 1 # Find out why only one thread
opt.batchSize=1# Why are these loosly here! They should be in the config file!
# Create a parameter for specifying whether we should be shuffling the data
# Dont perform any hectic data augmentation!

data_loader=DataLoader(opt)
dataset=data_loader.load()
# Dont create another model, just take the model that we already have!

print(len(dataset))
for i,data in enumerate(dataset):
    model.perform_update(data)# Inside here we setting the input
    model.predict()
    # Why would I be wanting the paths?  The Names!
    save the images!
