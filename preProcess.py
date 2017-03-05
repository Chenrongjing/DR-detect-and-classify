import cv2
import numpy

def preProcessImage(img):
	x,y,w,h = [423, 60, 1376, 1376]
	
	img=img[y:y+h,x:x+w]
	#imgLAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
	#imgLAB[:,:,0]= clahe.apply(imgLAB[:,:,0])
	
	#img=cv2.cvtColor(imgLAB,cv2.COLOR_LAB2BGR)

	#img[:,:,0] = clahe.apply(img[:,:,0])
	#img[:,:,1] = clahe.apply(img[:,:,1])
	#img[:,:,2] = clahe.apply(img[:,:,2])	

	img=cv2.resize(img, (512,512))
	imgThresh=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	__,imgThresh=cv2.threshold(imgThresh,20,255,cv2.THRESH_BINARY)
	
	img=cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

	mean,std=cv2.meanStdDev(img,imgThresh)
	
	img[:,:,0]=(img[:,:,0]-mean[0])/std[0]
	img[:,:,1]=(img[:,:,1]-mean[1])/std[1]
	img[:,:,2]=(img[:,:,2]-mean[2])/std[2]
	
	return img
