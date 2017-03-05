import glob
import cv2
import numpy
import numpy.random
import preProcess

from pyexcel_xls import get_data

def readDatabase(rootDirectory):
	numImages=0

	imageList=[]
	targetList=[]

	print(rootDirectory)
	xlsName = glob.glob(rootDirectory+"/*.xls")
	data = get_data(xlsName[0])
	xlsList=list(data.items()[0][1][1:])

	i=0
	for row in xlsList:
		img = cv2.imread(rootDirectory+'/'+str(row[0]))
		img = preProcess.preProcessImage(img)
		
		imageList.append(img.astype(numpy.float32))
		if(row[2]==0):
			targetList.append([1,0,0,0])
		if(row[2]==1):
			targetList.append([0,1,0,0])
		if(row[2]==2):
			targetList.append([0,0,1,0])
		if(row[2]==3):
			targetList.append([0,0,0,1])
		i=i+1
	
	numImages = i

	imageList=numpy.reshape(imageList,(numImages,3,512,512))
	targetList=numpy.float32(targetList)
	targetList=numpy.reshape(targetList,(numImages,4))
	
	return [imageList,targetList]
