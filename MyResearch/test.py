from SetupTraining import *
from ManageData import DataLoader
import Networks
import glob
import PIL
from PIL import Image


def display_current_results(images,title,phase='train'): # Perfect
    the_title= title.split('.')
    for label,image in images.items():# .items() extracts the "packages" from the dictionary
        img_path=os.path.join(opt.img_dir,'NEW -%s %s.png'%(the_title[0],label))
        image_pil=Image.fromarray(image)
        image_pil.save(img_path)


opt = process(TestingSetup(DefaultSetup()))
# Find out why only one thread
data_loader=DataLoader(opt)
dataset=data_loader.load()
model=Networks.The_Model(opt)
file_names = [os.path.basename(x) for x in glob.glob(str(opt.data_source)+"/testA/*")]
#Still need abit of my init function because I need to create the arch. to load the weights!
# They dont create descriminators when testing
print(len(dataset))
for i,data in enumerate(dataset):
    #model.perform_update(data)# Inside here we setting the input
    model.set_input(data)
    model.predict()

    display_current_results(model.for_displaying_images(),file_names[i],phase='test')

    # Why would I be wanting the paths?  The Names!
    #save the images!
