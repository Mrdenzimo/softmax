# run this after running loadata.py

sm = SoftMax()
sm.fit(datapatch,labelpatch)


predict_train_labels = sm.predict(train_images)
print "accuracy for train set: ", sum(predict_train_labels == train_labels)/60000.


predict_labels = sm.predict(test_images)
print "accuracy for test set: ",sum(predict_labels == test_labels) / 10000.
