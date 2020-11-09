import cv2
import glob
import matplotlib.pyplot as plt

directory = "../testB/"
for filename in glob.glob(directory + str("/*.png")) or glob.glob(directory + str("/*.jpg")):
    im = plt.imread(filename)
    resized =cv2.resize(im, (512,512), interpolation = cv2.INTER_AREA)
    plt.imsave(filename,resized)

