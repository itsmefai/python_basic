import pandas as pd

df = pd.read_csv('/Users/fairuz30/Desktop/ Machinelearning/[repo]Machine_Learning_A-Z_All_Codes_and_Templates-master/3. Multiple Linear Regression/50_Startups.csv')

#Define dependent and independent variable
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x_1 = LabelEncoder()
x[:,-1] = le_x_1.fit_transform(x[:,-1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

#split into training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

#feature scalling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#creating ANN Model
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#Creating input layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))

#creating hidden layer
classifier.add(Dense(output_dim =3, init = 'uniform', activation = 'relu'))

#creatin output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compile ANN layers
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

#fit model to training data
classifier.fit(x_train,y_train,batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(x_test)







