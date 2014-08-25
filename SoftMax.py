# implementation of softmax model
# hardmaru 2014

import numpy as np
import scipy.optimize as opt
import time

class SoftMax ():
	"""
	Softmax classifier using l-bfgs-b optimizer.

	This classifier uses the softmax algorithm, which generalizes
	logistic regression.

	The training is done using the entire training set,
	rather than minipatch or SVD.

	This implementation uses numpy and scipy for holding training data and
	for for the optizier respectively.

	Parameters
	----------
	reg_lambda: float, default 1e-4
		Regularization parameter to prevent overfitting

	maxiter: integer, default 200
		maximum number of iterations used for l-bfgs-b optimizer

	tolerance: float, default 1e=4
		early exit for optimizer

	disp: boolean, default = True
		set to false if no debug/progress info is to be displayed.

	data_input:  npArray of dimensions m training examples x n data points

	target_labels:  npArray of dimensions m labels x 1,
		corresponding to data_imput
		The labels can be anything, including strings, integers, floats

	method:  "l-bfgs-b" or "sgd" (default is "sgd", stochastic gradient descent w/ minibatch)

	Example Usage
	-------------

	# the training data must be of dimensions m x n,
	# and train_label of dimensions m x 1
	# m = number of training examples
	# n = size of each training example

	train_data, train_label = obtain_data("train_data.csv")
	test_data, test_label = obtain_data("test_data.csv")

	# create the softmax model
	# can finetune by setting reg_lambda, maxiter,
	# tolerance, disp in the init.

	sm = SoftMax()
	sm.fit(train_data, train_label)
	predict_label = sm.predict(test_data)

	# predict_label is the predicted set of labels from the training
	# it can be compared to test_label for accuracy

	References:
	-----------

	http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression

	"""

	def __init__(self, reg_lambda = 1e-4, maxiter = 400, tolerance = 1e-4, disp = True, method = "sgd"):
		self.reg_lambda = reg_lambda
		self.maxiter = maxiter
		self.disp = disp
		self.tolerance = 1e-9
		self.theta_unroll = 0
		self.has_trained_once = False
		self.method = method

	def predict(self, data):
		# test for trained theta and assert.
		m, k = np.shape(data)
		theta = np.reshape(self.theta_unroll,(k, self.num_classes))

		M = data.dot(theta)
		M_max = M.max(axis=1)
		M = np.transpose(M) - M_max
		M = np.exp(M)
		M /= M.sum(0)
		predict_raw = np.argmax(M, axis=0)
		prob = np.max(M,axis=0)
		predict = np.zeros(m, dtype=self.predict_dtype)
		for i in range(0,m):
			predict[i] = self.reverse_map_label[predict_raw[i]]
		return predict, prob

	def fit(self, data_input, target_labels_input):
		m_train = np.shape(data_input)[0]
		m_minibatch = 50 # if size is below minimatch, then no need to randomize.

		self._fit(data_input,target_labels_input, self.method)

	def _fit(self, data_input, target_labels_input, method = "l-bfgs-b"):

		data = np.array(data_input)
		target_labels = np.array(target_labels_input)

		self.labels = set(target_labels)
		num_classes = len(self.labels)
		self.predict_dtype = np.array(target_labels).dtype
		self.num_classes = num_classes
		self.map_label = dict(zip(self.labels,range(0,num_classes)))
		self.reverse_map_label = dict(zip(range(0,num_classes),self.labels))

		# m number of training examples, each having k features
		m, k = np.shape(data)
		self.num_features = k

		self._initialize_theta()

		label_raw = np.zeros(m, dtype='int') # integer label conversion
		for i in range(0,m):
			label_raw[i] = self.map_label[target_labels[i]]

		gt = np.zeros((m, num_classes), dtype='float') # ground truth matrix
		for i in range(0,m):
			gt[i,label_raw[i]] = 1.

		# example call:
		#cost, grad = self._cost(self.theta_unroll, num_classes, self.reg_lambda, data, gt)

		func = lambda x: self._cost(x, num_classes, self.reg_lambda, data, gt)

		# example call:
		# cost, grad = func(self.theta_unroll)
		#compare_numerical_gradient(func,self.theta_unroll)
		
		#opts = {'maxiter': self.maxiter, 'disp': self.disp}
		#res = opt.minimize(func,self.theta_unroll,method='L-BFGS-B', tol = self.tolerance, options=opts)
		
		if ( method == "l-bfgs-b" ):
			res = opt.fmin_l_bfgs_b(func,self.theta_unroll,pgtol=self.tolerance,disp=self.disp,maxiter=self.maxiter)
			if self.disp:
				print "---results---"
				print "cost at min point:",res[1] 
				print "theta at min point:",res[0]
				print "other info"
				print res[2]
				print "-------------"
				self.theta_unroll = np.array(res[0], dtype='float')
		else: # use stochastic gradient descent:
			func_sgd = lambda x: self._cost(x, num_classes, self.reg_lambda, data_batch, gt_batch)
			# randomise data order
			idx = np.random.randint(0,m,m)
			data = data[idx,:]
			gt = gt[idx,:]
			# split data/label batch into train set and cv set using 80/20 split
			num_epoch = 10
			alpha = 1.0
			theta = self.theta_unroll
			m_minibatch = 100
			num_batch = int(m/m_minibatch)
			for j in range(0,num_epoch):
				for i in range(0,num_batch):
					data_batch = data[i*m_minibatch:(i+1)*m_minibatch]
					gt_batch = gt[i*m_minibatch:(i+1)*m_minibatch]
					cost, grad = func_sgd(theta)
					theta = theta - alpha * grad
				alpha = alpha / 2.0
				if (self.disp):
					print "epoch #%d\t cost = %f\n" % (j, cost)			
			self.theta_unroll = theta
			cost, grad = func(theta)
			if self.disp:
				print "final cost = ",cost



	def _initialize_theta(self):
		# use sqrt(6)/sqrt(input+output+1 as selection)
		# only initialize if theta has not been trained yet.
		if (self.has_trained_once == False or len(self.theta_unroll) != (self.num_classes * self.num_features)):
			self.has_trained_once = True
			r = np.sqrt(6.)/np.sqrt(self.num_features + self.num_classes + 1)
			theta_unroll = np.random.rand(self.num_classes * self.num_features) * 2 * r - r
			self.theta_unroll = np.array(theta_unroll, dtype='float')

	def _cost(self, theta_unroll, num_classes, reg_lambda, data, gt):
		# softmax internal cost function
		
		m, k = np.shape(data)
		theta = np.reshape(theta_unroll,(k, num_classes))
		M = data.dot(theta)
		M_max = M.max(axis=1)
		M = np.transpose(M) - M_max
		M = np.exp(M)
		M /= M.sum(0)
		M = np.transpose(M)
		cost = np.sum(np.log(M)*gt) / (-1. * m) + np.sum(theta*theta)*reg_lambda*0.5
		grad = np.transpose(data).dot(gt-M) / (-1. * m) + reg_lambda * theta
		grad = np.reshape(grad, k*num_classes)

		return cost, grad

def compare_numerical_gradient(func, theta, epsilon = 1e-6):
	theta = np.array(theta, dtype='float')
	cost, grad = func(theta)
	numgrad = np.zeros(np.shape(grad))
	k = len(theta)

	for i in range(0,k):
		thetaplus = np.array(theta, dtype='float')
		thetaminus = np.array(theta, dtype='float')
		thetaplus[i] += epsilon
		thetaminus[i] -= epsilon
		up, grad2 = func(thetaplus)
		down, grad2 = func(thetaminus)
		numgrad[i] = (up - down)
		#print up, down, (up-down), numgrad[i]
	numgrad /= (2*epsilon)

	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(grad+numgrad)
	print grad
	print numgrad
	print "diff:",diff


