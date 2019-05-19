import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import cv2

def saveWeights1(le1, le2, le3, ld3, ld2, ld1):
	np.save("SRWB_2X_1/le1.npy", le1)
	np.save("SRWB_2X_1/le2.npy", le2)
	np.save("SRWB_2X_1/le3.npy", le3)
	np.save("SRWB_2X_1/ld3.npy", ld3)
	np.save("SRWB_2X_1/ld2.npy", ld2)
	np.save("SRWB_2X_1/ld1.npy", ld1)

def saveBiases1(be1, be2, be3, bd3, bd2, bd1):
	np.save("SRWB_2X_1/be1.npy", be1)
	np.save("SRWB_2X_1/be2.npy", be2)
	np.save("SRWB_2X_1/be3.npy", be3)
	np.save("SRWB_2X_1/bd3.npy", bd3)
	np.save("SRWB_2X_1/bd2.npy", bd2)
	np.save("SRWB_2X_1/bd1.npy", bd1)

def saveWeights2(le1, le2, le3, ld3, ld2, ld1):
	np.save("SRWB_2X_2/le1.npy", le1)
	np.save("SRWB_2X_2/le2.npy", le2)
	np.save("SRWB_2X_2/le3.npy", le3)
	np.save("SRWB_2X_2/ld3.npy", ld3)
	np.save("SRWB_2X_2/ld2.npy", ld2)
	np.save("SRWB_2X_2/ld1.npy", ld1)

def saveBiases2(be1, be2, be3, bd3, bd2, bd1):
	np.save("SRWB_2X_2/be1.npy", be1)
	np.save("SRWB_2X_2/be2.npy", be2)
	np.save("SRWB_2X_2/be3.npy", be3)
	np.save("SRWB_2X_2/bd3.npy", bd3)
	np.save("SRWB_2X_2/bd2.npy", bd2)
	np.save("SRWB_2X_2/bd1.npy", bd1)


ile1 = np.load("SRWB_2X_1/le1.npy")
ile2 = np.load("SRWB_2X_1/le2.npy")
ile3 = np.load("SRWB_2X_1/le3.npy")
ile4 = np.load("SRWB_2X_1/le4.npy")
ile5 = np.load("SRWB_2X_1/le5.npy")
ild5 = np.load("SRWB_2X_1/ld5.npy")
ild4 = np.load("SRWB_2X_1/ld4.npy")
ild3 = np.load("SRWB_2X_1/ld3.npy")
ild2 = np.load("SRWB_2X_1/ld2.npy")
ild1 = np.load("SRWB_2X_1/ld1.npy")

ibe1 = np.load("SRWB_2X_1/be1.npy")
ibe2 = np.load("SRWB_2X_1/be2.npy")
ibe3 = np.load("SRWB_2X_1/be3.npy")
ibe4 = np.load("SRWB_2X_1/be4.npy")
ibe5 = np.load("SRWB_2X_1/be5.npy")
ibd5 = np.load("SRWB_2X_1/bd5.npy")
ibd4 = np.load("SRWB_2X_1/bd4.npy")
ibd3 = np.load("SRWB_2X_1/bd3.npy")
ibd2 = np.load("SRWB_2X_1/bd2.npy")
ibd1 = np.load("SRWB_2X_1/bd1.npy")


sle1 = np.load("SRWB_2X_2/le1.npy")
sle2 = np.load("SRWB_2X_2/le2.npy")
sle3 = np.load("SRWB_2X_2/le3.npy")
sld3 = np.load("SRWB_2X_2/ld3.npy")
sld2 = np.load("SRWB_2X_2/ld2.npy")
sld1 = np.load("SRWB_2X_2/ld1.npy")

sbe1 = np.load("SRWB_2X_2/be1.npy")
sbe2 = np.load("SRWB_2X_2/be2.npy")
sbe3 = np.load("SRWB_2X_2/be3.npy")
sbd3 = np.load("SRWB_2X_2/bd3.npy")
sbd2 = np.load("SRWB_2X_2/bd2.npy")
sbd1 = np.load("SRWB_2X_2/bd1.npy")





def normalize(omax, omin, nmax, nmin, ip):
	return (nmax - nmin)/(omax-omin)*(ip-omax)+nmax

dirsave = "Res-1/"
dirdata = "Data/"
learning_rate=0.0001
epochs = 10000
batchsize = 10
display_step = 10

dimension = 128
n_input = dimension
patch_dimension = 64
n_output = n_input
dim=n_input


ll = 0
hl = 0
incr = batchsize

images = np.load(dirdata+'testSR_2X.npy')
widthofimages = np.load(dirdata+'width_testimages_2X.npy')
heightofimages = np.load(dirdata+'height_testimages_2X.npy')
countperimage = np.load(dirdata+'countperimage_2X.npy')
names = np.load(dirdata+'names_2X.npy')

images = images.astype(float)
images = normalize(255.0, 0.0, 1.0, 0.0, images)
totalsize=images.shape[0]
lowerlimit = 0
higherlimit = 0
def takeAllPatches(image, width, height):
	global patch_dimension
	global lim
	cnt = 0
	i = 0
	recreatedimage = np.zeros((height,width))
	image_array = image
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
				#print 'if-4'
		
			image_toshow = image_array
			recreatedimage[rs:re, cs:ce] = image_toshow[cnt]
			#print cropimage.shape
			cnt = cnt+1
			
			
			j=j+patch_dimension
		i=i+patch_dimension
	return recreatedimage


	
#WEIGHTS AND BIASES

n1 = 32
n2 = 16
n3 = 16
n4 = 8
n5 = 8


ksize1= 5

weightsRED1 = {
	'ce1' : tf.Variable(ile1),
	'ce2' : tf.Variable(ile2),
	'ce3' : tf.Variable(ile3),
	'ce4' : tf.Variable(ile4),
	'ce5' : tf.Variable(ile5),
	'cd5' : tf.Variable(ild5),
	'cd4' : tf.Variable(ild4),
	'cd3' : tf.Variable(ild3),
	'cd2' : tf.Variable(ild2),
	'cd1' : tf.Variable(ild1)
}

biasesRED1 = {
	'be1' : tf.Variable(ibe1),
	'be2' : tf.Variable(ibe2),
	'be3' : tf.Variable(ibe3),
	'be4' : tf.Variable(ibe4),
	'be5' : tf.Variable(ibe5),
	'bd5' : tf.Variable(ibd5),
	'bd4' : tf.Variable(ibd4),
	'bd3' : tf.Variable(ibd3),
	'bd2' : tf.Variable(ibd2),
	'bd1' : tf.Variable(ibd1)
}

weightsUS1 = {
	'ce1' : tf.Variable(sle1),
	'ce2' : tf.Variable(sle2),
	'ce3' : tf.Variable(sle3),
	'cd3' : tf.Variable(sld3),
	'cd2' : tf.Variable(sld2),
	'cd1' : tf.Variable(sld1)
}

biasesUS1 = {
	'be1' : tf.Variable(sbe1),
	'be2' : tf.Variable(sbe2),
	'be3' : tf.Variable(sbe3),
	'bd3' : tf.Variable(sbd3),
	'bd2' : tf.Variable(sbd2),
	'bd1' : tf.Variable(sbd1)
}
weightsRED2 = {
	'ce1' : tf.Variable(ile1),
	'ce2' : tf.Variable(ile2),
	'ce3' : tf.Variable(ile3),
	'ce4' : tf.Variable(ile4),
	'ce5' : tf.Variable(ile5),
	'cd5' : tf.Variable(ild5),
	'cd4' : tf.Variable(ild4),
	'cd3' : tf.Variable(ild3),
	'cd2' : tf.Variable(ild2),
	'cd1' : tf.Variable(ild1)
}

biasesRED2 = {
	'be1' : tf.Variable(ibe1),
	'be2' : tf.Variable(ibe2),
	'be3' : tf.Variable(ibe3),
	'be4' : tf.Variable(ibe4),
	'be5' : tf.Variable(ibe5),
	'bd5' : tf.Variable(ibd5),
	'bd4' : tf.Variable(ibd4),
	'bd3' : tf.Variable(ibd3),
	'bd2' : tf.Variable(ibd2),
	'bd1' : tf.Variable(ibd1)
}

weightsUS2 = {
	'ce1' : tf.Variable(sle1),
	'ce2' : tf.Variable(sle2),
	'ce3' : tf.Variable(sle3),
	'cd3' : tf.Variable(sld3),
	'cd2' : tf.Variable(sld2),
	'cd1' : tf.Variable(sld1)
}

biasesUS2 = {
	'be1' : tf.Variable(sbe1),
	'be2' : tf.Variable(sbe2),
	'be3' : tf.Variable(sbe3),
	'bd3' : tf.Variable(sbd3),
	'bd2' : tf.Variable(sbd2),
	'bd1' : tf.Variable(sbd1)
}

weightsRED3 = {
	'ce1' : tf.Variable(ile1),
	'ce2' : tf.Variable(ile2),
	'ce3' : tf.Variable(ile3),
	'ce4' : tf.Variable(ile4),
	'ce5' : tf.Variable(ile5),
	'cd5' : tf.Variable(ild5),
	'cd4' : tf.Variable(ild4),
	'cd3' : tf.Variable(ild3),
	'cd2' : tf.Variable(ild2),
	'cd1' : tf.Variable(ild1)
}

biasesRED3 = {
	'be1' : tf.Variable(ibe1),
	'be2' : tf.Variable(ibe2),
	'be3' : tf.Variable(ibe3),
	'be4' : tf.Variable(ibe4),
	'be5' : tf.Variable(ibe5),
	'bd5' : tf.Variable(ibd5),
	'bd4' : tf.Variable(ibd4),
	'bd3' : tf.Variable(ibd3),
	'bd2' : tf.Variable(ibd2),
	'bd1' : tf.Variable(ibd1)
}

weightsUS3 = {
	'ce1' : tf.Variable(sle1),
	'ce2' : tf.Variable(sle2),
	'ce3' : tf.Variable(sle3),
	'cd3' : tf.Variable(sld3),
	'cd2' : tf.Variable(sld2),
	'cd1' : tf.Variable(sld1)
}

biasesUS3 = {
	'be1' : tf.Variable(sbe1),
	'be2' : tf.Variable(sbe2),
	'be3' : tf.Variable(sbe3),
	'bd3' : tf.Variable(sbd3),
	'bd2' : tf.Variable(sbd2),
	'bd1' : tf.Variable(sbd1)
}





def leaky_rrelu(x, alpha=0.2):
	return tf.maximum(x, alpha*x)


def caeRED(_X, _W, _b, _keepprob, alpha = 0.2):
	_input_r = _X
	#ENCODER
	_le1 = tf.add(tf.nn.conv2d(_input_r, _W['ce1'], strides = [1,1,1,1], padding='SAME'), _b['be1'])
	_ce1 = tf.nn.relu(_le1)
	_ce1 = tf.nn.dropout(_ce1, _keepprob)

	_le2 = tf.add(tf.nn.conv2d(_ce1, _W['ce2'], strides = [1,1,1,1], padding='SAME'), _b['be2'])
	_ce2 = tf.nn.relu(_le2)
	_ce2 = tf.nn.dropout(_ce2, _keepprob)

	_le3 = tf.add(tf.nn.conv2d(_ce2, _W['ce3'], strides = [1,1,1,1], padding='SAME'), _b['be3'])
	_ce3 = tf.nn.relu(_le3)
	_ce3 = tf.nn.dropout(_ce3, _keepprob)

	_le4 = tf.add(tf.nn.conv2d(_ce3, _W['ce4'], strides = [1,1,1,1], padding='SAME'), _b['be4'])
	_ce4 = tf.nn.relu(_le4)
	_ce4 = tf.nn.dropout(_ce4, _keepprob)

	_le5 = tf.add(tf.nn.conv2d(_ce4, _W['ce5'], strides = [1,1,1,1], padding='SAME'), _b['be5'])
	_ce5 = tf.nn.relu(_le5)
	_ce5 = tf.nn.dropout(_ce5, _keepprob)

	_ld5 = tf.add(tf.nn.conv2d_transpose(_ce5, _W['cd5'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1],tf.shape(_input_r)[2],n4]), strides = [1,1,1,1], padding = 'SAME'), _b['bd5'])
	_ld5 = _ld5 + _le4
	_cd5 = tf.nn.relu(_ld5)
	_cd5 = tf.nn.dropout(_cd5, _keepprob)

	_ld4 = tf.add(tf.nn.conv2d_transpose(_cd5, _W['cd4'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1],tf.shape(_input_r)[2],n3]), strides = [1,1,1,1], padding = 'SAME'), _b['bd4'])
	_ld4 = _ld4 + _le3
	_cd4 = tf.nn.relu(_ld4)
	_cd4 = tf.nn.dropout(_cd4, _keepprob)

	_ld3 = tf.add(tf.nn.conv2d_transpose(_cd4, _W['cd3'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1],tf.shape(_input_r)[2],n2]), strides = [1,1,1,1], padding = 'SAME'), _b['bd3'])
	_ld3 = _ld3 + _le2
	_cd3 = tf.nn.relu(_ld3)
	_cd3 = tf.nn.dropout(_cd3, _keepprob)

	_ld2 = tf.add(tf.nn.conv2d_transpose(_cd3, _W['cd2'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1],tf.shape(_input_r)[2],n1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd2'])
	_ld2 = _ld2 + _le1	
	_cd2 = tf.nn.relu(_ld2)
	_cd2 = tf.nn.dropout(_cd2, _keepprob)

	_ld1 = tf.add(tf.nn.conv2d_transpose(_cd2, _W['cd1'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1],tf.shape(_input_r)[2],1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd1'])	
	_ld1 = _ld1 + _X
	_cd1 = tf.nn.relu(_ld1)
	_cd1 = tf.nn.dropout(_cd1, _keepprob)	
	_out = _cd1

	return _out

def caeUS(_X, _W, _b, _keepprob, alpha = 0.2):
	_input_r = _X
	#ENCODER
	_le1 = tf.add(tf.nn.conv2d(_input_r, _W['ce1'], strides = [1,1,1,1], padding='SAME'), _b['be1'])
	_ce1 = tf.nn.relu(_le1)
	_ce1 = tf.nn.dropout(_ce1, _keepprob)

	_le2 = tf.add(tf.nn.conv2d(_ce1, _W['ce2'], strides = [1,1,1,1], padding='SAME'), _b['be2'])
	_ce2 = tf.nn.relu(_le2)
	_ce2 = tf.nn.dropout(_ce2, _keepprob)

	_le3 = tf.add(tf.nn.conv2d(_ce2, _W['ce3'], strides = [1,1,1,1], padding='SAME'), _b['be3'])
	_ce3 = tf.nn.relu(_le3)
	_ce3 = tf.nn.dropout(_ce3, _keepprob)

	

	_ld3 = tf.add(tf.nn.conv2d_transpose(_ce3, _W['cd3'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1]*2,tf.shape(_input_r)[2]*2,n2]), strides = [1,2,2,1], padding = 'SAME'), _b['bd3'])
	#_ld3 = _ld3 + _le2
	_cd3 = tf.nn.relu(_ld3)
	_cd3 = tf.nn.dropout(_cd3, _keepprob)

	_ld2 = tf.add(tf.nn.conv2d_transpose(_cd3, _W['cd2'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1]*2,tf.shape(_input_r)[2]*2,n1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd2'])
	#_ld2 = _ld2 + _le1	
	_cd2 = tf.nn.relu(_ld2)
	_cd2 = tf.nn.dropout(_cd2, _keepprob)

	_ld1 = tf.add(tf.nn.conv2d_transpose(_cd2, _W['cd1'], tf.stack([tf.shape(_X)[0],tf.shape(_input_r)[1]*2,tf.shape(_input_r)[2]*2,1]), strides = [1,1,1,1], padding = 'SAME'), _b['bd1'])	
	#_ld1 = _ld1 + _X
	_cd1 = tf.nn.relu(_ld1)
	_cd1 = tf.nn.dropout(_cd1, _keepprob)	
	_out = _cd1

	return _out


def calculateL2loss(im1, im2):
	return tf.reduce_mean(tf.square(im1-im2))

def calculateL1loss(im1, im2):
	return tf.reduce_sum(tf.abs(im1-im2))

def optimize(cost, learning_rate = 0.0001):
	return tf.train.AdamOptimizer(learning_rate).minimize(cost)


print ("Network ready")

x = tf.placeholder(tf.float32, [None, None, None, 1])
y = tf.placeholder(tf.float32, [None, None, None, 1])

keepprob = tf.placeholder(tf.float32)

predUS1 = caeUS(x, weightsUS1, biasesUS1, keepprob)
predRED1 = caeRED(predUS1, weightsRED1, biasesRED1, keepprob)

predUS2 = caeUS(predRED1, weightsUS2, biasesUS2, keepprob)
predRED2 = caeRED(predUS2, weightsRED2, biasesRED2, keepprob)

predUS3 = caeUS(predRED2, weightsUS3, biasesUS3, keepprob)
predRED3 = caeRED(predUS3, weightsRED3, biasesRED2, keepprob)





cost1 = calculateL2loss(predRED3, y)


optm1 = optimize(cost1, learning_rate)





init = tf.global_variables_initializer()

print("All functions ready")
saver = tf.train.Saver()

print ("Start testing")

with tf.Session() as sess:
	saver.restore(sess, "logs/SR/8X_fs1.ckpt")	
	for i in range(countperimage.shape[0]):
		higherlimit = int(higherlimit+countperimage[i])
		allpatchesofanimage = images[lowerlimit:higherlimit].copy()
		lowerlimit = int(lowerlimit + countperimage[i])
		reconstructedimage = takeAllPatches(allpatchesofanimage, int(widthofimages[i]), int(heightofimages[i]))
		recon = sess.run(predRED3, feed_dict = {x:reconstructedimage.reshape(1, reconstructedimage.shape[0], reconstructedimage.shape[1], 1), keepprob:1.})
		recreatedimage = recon.reshape(reconstructedimage.shape[0]*8, reconstructedimage.shape[1]*8)
		recreatedimage = normalize(1.0, 0.0, 255.0, 0.0, recreatedimage)
		cv2.imwrite(dirsave+names[i], recreatedimage)

	lowerlimit=0
	higherlimit=0

		


		
		









































