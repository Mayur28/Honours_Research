import torch.utils.data as data
def DataLoader(opt):
    return CustomDataLoader().initialize(opt)

def MakeDataset(opt):
    return FullDataset().initialize(opt)

class CustomDataLoader:
    def __init__(self):
        pass
    
    
    def initialize(opt):
        self.opt=opt
        self.dataset=MakeDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=opt.batchSize,shuffle= True, num_workers=int(opt.nThreads))
        
    
    def load(self):# This will return the iterable over the dataset
        return self.dataloader
    
    #This function is compulsory when creating custom dataloaders!
    def __len__(self):
        return len(self.dataset)
        
    #Try to implement a __getitem__ to extract individual instances as well!
        
        
class FullDataset(data.Dataset):
    def __init__(self):
        super(FullDataset,self).__init__()
        
    def initialize(self,opt):
        #Check if we really need anything from Base_dataset!
        self.opt=opt
        #Form the path's to the data
        self.dir_A=os.path.join('../final_dataset',opt.phase+'A')
        self.dir_B=os.path.join('../final_dataset',opt.phase+'B')
        print("Path of A (dark images)"+str(self.dir_A))
        print("Path of B (Light images)"+str(self.dir_B))

        
        