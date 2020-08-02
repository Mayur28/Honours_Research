from SetupTraining import SetupTraining
from ManageData import DataLoader
import Networks



opt=SetupTraining().parse()# This is obviously just a start, I can (and must) adjust it accordingly
#Just check if I really need config!
data_loader=DataLoader(opt)
dataset=data_loader.load()
print("Number of training Images: %d"% len(data_loader))


the_model=Networks.The_Model(opt)
