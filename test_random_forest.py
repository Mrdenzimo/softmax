# run this after running loadata.py

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)
rf.fit(datapatch,labelpatch)

predict_train_labels = rf.predict(train_images)
print "accuracy for train set: ", sum(predict_train_labels == train_labels)/60000.

predict_labels = rf.predict(test_images)
print "accuracy for test set: ",sum(predict_labels == test_labels) / 10000.
