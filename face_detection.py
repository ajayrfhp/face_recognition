import numpy as np 
import scipy.io
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm,metrics, preprocessing
data = np.transpose(np.array(scipy.io.loadmat('./data.mat')['C']))

labels = np.ones((data.shape[0],1))
for i in range(labels.shape[0]):
	labels[i] = i/10


data = preprocessing.scale(data)

print labels.shape
print data.shape



covariance_matrix = np.cov(data)
eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

z = zip(eig_values,eig_vectors)
z.sort(key = lambda x:x[0])
eig_values,eig_vectors = zip(*z)
eig_vectors = np.array(eig_vectors)
eig_values = np.array(eig_values)




plt.plot(eig_values)
plt.xlabel('Eigen vectors')
plt.ylabel('Eigen values')
plt.show()
eig_vectors = eig_vectors[::-1]

important_eig_vectors = eig_vectors[:,0:200]
transformed_data = np.dot(data,important_eig_vectors)
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size = 0.33,random_state=42)
clf = svm.SVC()
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
print metrics.accuracy_score(preds,y_test)

