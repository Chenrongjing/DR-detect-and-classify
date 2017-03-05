import numpy as np
import sys
import matplotlib.pyplot as plt

def main():

	outputDirectory=''

	if(len(sys.argv)>=2):
		print('Output Directory: '+str(sys.argv[1]))
		outputDirectory=sys.argv[1]
	
	trainingLoss=np.float32([])
	trainingPrecision=np.float32([])
	validationLoss=np.float32([])
	validationPrecision=np.float32([])

	trainingLoss = np.memmap(outputDirectory+'/trainingLoss.dat', dtype='float32', mode='r')
	trainingPrecision = np.memmap(outputDirectory+'/trainingPrecision.dat', dtype='float32', mode='r')
	validationLoss = np.memmap(outputDirectory+'/validationLoss.dat', dtype='float32', mode='r')
	validationPrecision = np.memmap(outputDirectory+'/validationPrecision.dat', dtype='float32', mode='r')

	#plot
	index=np.arange(1,np.size(trainingLoss)+1,1)
	
	plt.plot(index,trainingLoss,'r',index,validationLoss,'b')
	plt.ylabel('Training Loss and Validation Loss')
	plt.show()

	plt.figure()
	plt.plot(index,trainingPrecision,'r',index,validationPrecision,'b')
	plt.ylabel('Training Accuracy and Validation Accuracy')
	plt.show()

	print((trainingLoss,trainingPrecision,validationLoss,validationPrecision))
	print(np.size(trainingLoss))

if __name__ == '__main__':
	main()
