from SetupTraining import *
from ManageData import DataLoader


opt=TestSetup(DefaultSetup())
# Find out why only one thread
data_loader=DataLoader(opt)
dataset=data_loader.load()
#Still need abit of my init function because I need to create the arch. to load the weights!
# They dont create descriminators when testing
print(len(dataset))
for i,data in enumerate(dataset):
    print("HELLO")
    #model.perform_update(data)# Inside here we setting the input
    #model.predict()
    # Why would I be wanting the paths?  The Names!
    #save the images!
