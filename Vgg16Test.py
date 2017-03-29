from Vgg16 import *
from PIL import Image
import scipy.misc

model_path = "./model/vgg16.npy"
image1 = "./images/puzzle.jpeg"
image2 = "./images/tiger.jpeg"
image3 = "./images/terrierimage.jpg"

classIndexToLabelMapping = "./classIndxLabelMapping.txt"

imagePaths = []
imagePaths.append(image1)
imagePaths.append(image2)
imagePaths.append(image3)

def buildImageArray(imagePaths):
	images = []
	for imagePath in imagePaths:
		imageData = Image.open(imagePath)
		imageData.load()
		imageAsArray = np.asarray(imageData)
		imageAsArray = scipy.misc.imresize(imageAsArray,(224,224))
		images.append(imageAsArray)
	return np.array(images)

with tf.Session() as sess:
	images = buildImageArray(imagePaths)
	vgg16 = Vgg16(model_path)
	prob = vgg16.predict(images,sess)
	maxProbIndexes = sess.run(tf.argmax(prob[0],1))
	classIndexToSynsetMapping = [l.strip() for l in open(classIndexToLabelMapping).readlines()]
	for maxProbIndex in maxProbIndexes:
		print classIndexToSynsetMapping[maxProbIndex]
