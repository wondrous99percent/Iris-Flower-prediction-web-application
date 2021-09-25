import numpy as np
import pandas as pd
import pickle

df=pd.read_csv('iris1.txt')

x=np.array(df.iloc[:, 0:4])
y=np.array(df.iloc[:, 4])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y) #to convert the string values into binary i.e 0,1,2

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
sv=SVC(kernel='linear').fit(x_train,y_train)

pickle.dump(sv,open("iris.pkl", 'wb'))