import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import pinv2
from tqdm.notebook import tqdm

dataset = pd.read_csv('up.csv')
dataset = dataset[ dataset['YEAR']>1980 ]
#dataset = dataset[ dataset['SUBDIVISION'] != 'RAYALSEEMA']
#dataset = dataset[ dataset['SUBDIVISION'] != 'KUCH']
#dataset = dataset[ dataset['SUBDIVISION'] != 'LAKSHWADEEP']
#dataset = dataset[ dataset['SUBDIVISION'] != 'VIDARBHA']
dataset = dataset.dropna()

#without climate
X = dataset.iloc[:,[0,3,4,6]].values
y = dataset.iloc[:,5].values

dataset2 = dataset[dataset['SEVERITY']!=0]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
# Encoding the Dependent Variable


#For Neural network

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



#without subdivision
"""
X = dataset.iloc[:,[3,4,6,7]].values
y = dataset.iloc[:,5].values

dataset2 = dataset[dataset['SEVERITY']>0]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 
"""

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = np.reshape(y_train,(-1,1))
y_train = onehotencoder.fit_transform(y_train).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train.shape,X_train)



input_size = X_train.shape[1]


hidden_size = 1000
input_weights = np.random.normal(size=[input_size,hidden_size])
biases = np.random.normal(size=[hidden_size])



def relu(x):
    return np.maximum(x, 0, x)


def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

output_weights = np.dot(pinv2(hidden_nodes(X_train)), y_train)


def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)
correct = 0
total = X_test.shape[0]

for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_test[i])
    correct += 1 if predicted == actual else 0
accuracy = correct/total
print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)


