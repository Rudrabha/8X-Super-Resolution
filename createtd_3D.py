import cv2
import numpy as np
import glob
import os


patch_dimension = 128
MAXSIZE = 300000
totalarray = np.zeros([MAXSIZE, patch_dimension, patch_dimension, 3])
patch_size = patch_dimension*patch_dimension
totalpatches = 0


#imagedir = "/home/manoj/OCR_train"
imagedir = "Res-2"
dirsave = "Data/"
totalcnt = 0
harray=[]
warray=[]
countperimage = []
names = []
lim=0
def getPatches(image, height, width):
	i=0
	cnt=0
	global lim
	global totalarray
	while (i<height):
		j=0
		while (j<width):

			if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
				rs=i
				re = i+patch_dimension
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension >= height and j+patch_dimension <=width-1:
				rs = height-(patch_dimension)
				re = height
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension <= height-1 and j+patch_dimension >=width:
				rs = i
				re = i+patch_dimension
				cs = width - (patch_dimension)
				ce = width

			if i+patch_dimension >= height and j+patch_dimension >=width:
				rs = height-(patch_dimension)
				re = height
				cs = width - (patch_dimension)
				ce = width

		
			cropimage = image[rs:re, cs:ce]
			cnt = cnt+1
			temparray = cropimage
			totalarray[lim] = temparray
			lim = lim+1
			j=j+patch_dimension
		i=i+patch_dimension
	return cnt
		
for img_from_folder in sorted(glob.glob(imagedir+"/*.png")):
	img = cv2.imread(img_from_folder)
	fname=img_from_folder
	fname = os.path.basename(fname)
	print fname
	#img = cv2.resize(img, (96,48))
	height = img.shape[0]
	harray = np.append(harray, height)
	width = img.shape[1]
	warray = np.append(warray, width)
	names = np.append(names, fname)
	#print img.shape
	count = getPatches(img, height,width)


	countperimage = np.append(countperimage, count)
	totalcnt += count
totalarray=totalarray[0:lim]
print totalarray.shape
np.save(dirsave+'testSR_2X.npy', totalarray)
np.save(dirsave+'names_2X.npy', names)
np.save(dirsave+'width_testimages_2X.npy', warray)
np.save(dirsave+"height_testimages_2X.npy", harray)
np.save(dirsave+'countperimage_2X.npy', countperimage)
print totalcnt
