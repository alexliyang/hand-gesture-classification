import h5py    
import numpy as np

''' read the hdf5 files '''
hf = h5py.File('hog_training_features.h5', 'r')
X_train=hf['trainingFeatures'][:].transpose() # transpose array

hf = h5py.File('hog_training_labels.h5', 'r')
Y_train=hf['trainingLabels'][:]

hf = h5py.File('hog_test_features.h5', 'r')
X_test=hf['testFeatures'][:].transpose() # transpose array

hf = h5py.File('hog_test_Labels.h5', 'r')
Y_test=hf['testLables'][:]

#print(list(hf.values()))

print (X_train.shape)
print (Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


''' support vector machine classifier '''
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, Y_train)
svm_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

acc = accuracy_score(Y_test, svm_pred)
cnf = confusion_matrix(Y_test, svm_pred)
print(acc)
print(cnf)

# plot the ROC characteristics
fpr, tpr, _ = roc_curve(Y_test, svm_pred, pos_label = 1)

plt.plot(fpr, tpr)
plt.show()


''' gaussian naive bayes classifier '''
from sklearn.naive_bayes import GaussianNB # use Gaussian Naive Bayes
nbc = GaussianNB()
nbc.fit(X_train, Y_train)
nbc_pred=nbc.predict(X_test)

acc = accuracy_score(Y_test, nbc_pred)
cnf = confusion_matrix(Y_test, nbc_pred)
print(acc)
print(cnf)

# plot the ROC characteristics
fpr, tpr, _ = roc_curve(Y_test, nbc_pred, pos_label = 1)

plt.plot(fpr, tpr)
plt.show()


''' k-nearest neighbors classifier'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
knn_pred=knn.predict(X_test)

acc = accuracy_score(Y_test, knn_pred)
cnf = confusion_matrix(Y_test, knn_pred)
print(acc)
print(cnf)

# plot the ROC characteristics
fpr, tpr, _ = roc_curve(Y_test, knn_pred, pos_label = 1)

plt.plot(fpr, tpr)
plt.show()

''' perform 5-fold cross validation on SVM '''
from sklearn.model_selection import cross_val_score
svm_scores = cross_val_score(clf, X_train, Y_train, cv = 5)

# report the accuracy of the model
print("SVM Accuracy: %0.2f (+/- %0.2f)" % (svm_scores.mean(), svm_scores.std() * 2))
