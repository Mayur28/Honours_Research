from Setup import *
from ManageData import DataLoader
import Networks
import glob
import PIL
from PIL import Image


def save_images(images,title,phase='train'):
    the_title= title.split('.')
    for label,image in images.items():# .items() extracts the "packages" from the dictionary
        img_path=os.path.join(opt.img_dir,'%s_%s-Enh.png'%(the_title[0],label)) #
        image_pil=Image.fromarray(image)
        image_pil.save(img_path)


opt = process(TestingSetup(DefaultSetup()))
data_loader=DataLoader(opt)
dataset=data_loader.load()
model=Networks.The_Model(opt)
file_names = [os.path.basename(x) for x in glob.glob(str(opt.data_source)+"/testA/*")]
print(len(dataset))
for i,data in enumerate(dataset):
    model.set_input(data)
    model.predict()
    print("Processing: "+str(file_names[i]))
    save_images(model.for_displaying_images(),file_names[i],phase='test')
