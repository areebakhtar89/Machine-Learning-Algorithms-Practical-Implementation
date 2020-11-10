import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset('iris')
print(df)
#sns.set()
#sns.pairplot(df, hue='species', size=1.5)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
print('------------Applying Naive Bayes Classifier-------------')
classifier_NB = GaussianNB()
classifier_NB.fit(X_train,Y_train)

print('------------K Nearest Neighbor Classifier-------------')
classifier_KNN = KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train,Y_train)

print('------------Decision Tree Classifier-------------')
classifier_DT = DecisionTreeClassifier(criterion='entropy')
classifier_DT.fit(X_train,Y_train)

print('------------Random Forest Classifier-------------')
classifier_RFC = RandomForestClassifier(n_estimators=10, criterion='gini')
classifier_RFC.fit(X_train,Y_train)

prediction_KNN = classifier_KNN.predict(X_test)
prediction_NB = classifier_NB.predict(X_test)
prediction_DT = classifier_DT.predict(X_test)
prediction_RFC = classifier_RFC.predict(X_test)
print('-----------------------predicted of KNN-----------------------')
print(prediction_KNN)
print('-----------------------predicted of NB-----------------------')
print(prediction_NB)
print('-----------------------predicted of DT-----------------------')
print(prediction_DT)
print('-----------------------predicted of RFC-----------------------')
print(prediction_RFC)
print('----------------------Actual----------------------------')
print(Y_test)
#manual method for calculating accuracy
#Y_test_df=pd.DataFrame(Y_test)

#Y_test_df = Y_test_df.iloc[:,:1].values
#Y_test_df = Y_test_df.reshape(len(Y_test_df),)
#prediction = prediction.astype('object')
#count=0
#for i in range(len(prediction)):
#    if (prediction[i]!=Y_test_df[i]):
#        count=count+1
        
#print(count)
#accuracy = ((len(X_test)-count)/len(X_test))
#print("manual accuracy: ",round(accuracy,2))

#direct method for calculating accuracy

print('System Computed Accuracy KNN:', accuracy_score(Y_test, prediction_KNN))
print('System Computed Accuracy NB:', accuracy_score(Y_test, prediction_NB))
print('System Computed Accuracy DT:', accuracy_score(Y_test, prediction_DT))
print('System Computed Accuracy RFC:', accuracy_score(Y_test, prediction_RFC))

mat_NB = confusion_matrix(Y_test, prediction_NB)
mat_KNN = confusion_matrix(Y_test, prediction_KNN)
mat_DT = confusion_matrix(Y_test, prediction_DT)
mat_RFC = confusion_matrix(Y_test, prediction_RFC)

print('NB\n', mat_NB)
print('KNN\n', mat_KNN)
print('DT\n', mat_DT)
print('RFC\n', mat_RFC)

sns.heatmap(mat_NB, square=True, annot=True, cbar=True)
plt.xlabel('predicted value NB')
plt.ylabel('actual value')
plt.show()

sns.heatmap(mat_KNN, square=True, annot=True, cbar=True)
plt.xlabel('predicted value KNN')
plt.ylabel('actual value')
plt.show()

sns.heatmap(mat_DT, square=True, annot=True, cbar=True)
plt.xlabel('predicted value DT')
plt.ylabel('actual value')
plt.show()

sns.heatmap(mat_RFC, square=True, annot=True, cbar=True)
plt.xlabel('predicted value RFC')
plt.ylabel('actual value')
plt.show()

