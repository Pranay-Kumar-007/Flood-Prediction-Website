from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np


dataset = pd.read_csv("up.csv")
dataset = dataset.dropna()
test_data = dataset.copy()
test_data.append(["ASSAM & MEGHALAYA",0,0,"Jun-Sep",1589.8,0,"Hilly"])
test_data.to_csv("up1.csv")
#print(df.head())
X = dataset.iloc[:,[0,3,4,6]].values
y = dataset.iloc[:,5].values
#print(X[0])
dataset2 = dataset[dataset['SEVERITY']!=0]

test_x = test_data.iloc[4000:,[0,3,4,6]].values
print(test_data.tail())

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



labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#print(y)

# Split data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
onehotencoder = OneHotEncoder(categorical_features = [0])
#y_train = np.reshape(y_train,(-1,1))
#y_train = onehotencoder.fit_transform(y_train).toarray()
#print(X_train.shape)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

pred = dt.predict(X_test)
dot_data = StringIO()
li = []
for i in range(46):
    li.append(str(i))

export_graphviz(dt, out_file=dot_data, feature_names=li)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



print(pred)
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, pred))



from sklearn.externals import joblib

joblib.dump(dt, 'filename.pkl') 

