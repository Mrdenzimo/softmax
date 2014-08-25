import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math as math

def display_data(data, label, size_x, size_y):
	m,k = np.shape(data)
	images = []
	for i in range(0,m):
		newdata = np.array(data[i,:].reshape(size_x,size_y))
		images.append(np.transpose(newdata))

	ncols = int(math.ceil(math.sqrt(m)))
	nrows = int(math.ceil(m/ncols))
	fig = plt.figure()
	fig.subplots_adjust(hspace=1.2,wspace=0.2)

	for i in range(0,m):
		a=fig.add_subplot(nrows,ncols,i)
		a.set_title(str(label[i])+':')
		plt.imshow(images[i], cmap = cm.Greys)
		plt.axis('off')



	#plt.tight_layout()
	plt.show()
	

	
