import glob
import cv2
import numpy
import numpy.random
import preProcess

from pyexcel_xls import get_data

def readDatabase(rootDirectory):
	numImage,weight0,weight1,weight2,weight3=(0,0,0,0,0)

	print(rootDirectory)
	xlsName = glob.glob(rootDirectory+"/*.xls")
	data = get_data(xlsName[0])
	xlsList=list(data.items()[0][1][1:])
	i=0
	for row in xlsList:
		if(row[2]==0):
			weight0=weight0+1
		if(row[2]==1):
			weight1=weight1+1
		if(row[2]==2):
			weight2=weight2+1
		if(row[2]==3):
			weight3=weight3+1
		numImage=numImage+1
	
	return numpy.array([numImage,weight0,weight1,weight2,weight3])

def main():

	databases = ['/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base11','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base12','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base13','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base14','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base21','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base22','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base23','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base24','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base31','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base32','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base33','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base34']

	numImage,weight0,weight1,weight2,weight3=[0,0,0,0,0]

	for index in range(0,12):
		print("Index:")
		print(index)
		numImage,weight0,weight1,weight2,weight3=readDatabase(databases[index])+numpy.array([numImage,weight0,weight1,weight2,weight3])
		print(readDatabase(databases[index]))
	print('Number of images, number of \'0\', number of \'1\', number of \'2\', number of \'3\':'+str((numImage,weight0,weight1,weight2,weight3)))
	weight0,weight1,weight2,weight3=numpy.array(numpy.float32((weight0,weight1,weight2,weight3)))/numImage
	weight0,weight1,weight2,weight3=0.25/numpy.array(numpy.float32((weight0,weight1,weight2,weight3)))
	print('class weight of \'0\', class weight of \'1\', class weight of \'2\', class weight of \'3\':'+str((weight0,weight1,weight2,weight3)))
	
if __name__ == '__main__':
	main()
