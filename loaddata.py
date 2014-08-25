import scipy.io as sio
import numpy as np
import numpy.random as random

filename = 'mnist.mat'

mat_data = sio.loadmat(filename)
train_images = mat_data['images'].transpose()
train_labels = mat_data['labels'][:,0]
test_images = mat_data['testImages'].transpose()
test_labels = mat_data['testLabels'][:,0]
m_train, k_train = np.shape(train_images)
m_test, k_test = np.shape(test_images)

N_choose = 60000
idx = random.randint(0,m_train,N_choose)
datapatch = train_images[idx,:]
labelpatch = train_labels[idx]

smalldata = datapatch[0:30,:]
smalllabel = labelpatch[0:30]

#display_data(datapatch,labelpatch,28,28)