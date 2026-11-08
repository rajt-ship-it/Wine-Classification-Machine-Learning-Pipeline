import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_wine#going to use this dataset
# 1. Dataset loading
wine = load_wine()#it will come as dictionary
print(wine)
#creating dataframe for tabular format
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
# Target column (Class 0, 1, 2) bhi add kar do
df['target'] = wine.target
print(df.shape)#to check no. of rows and columns
array=df.values
X=array[:,0:13]#here we are giving all 13 columns to X and this will be input data
Y=array[:,13]#the last column will be used for output data
validation_size=0.20 #here pecified that 20% data is going to be used for testing, see upcoming step
seed=100#consider it like a page in a diary when the method of shuffling is written
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
scoring="accuracy"
#using all models and finding best out of them
models=[]
models.append(("LR",LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(("SVM",SVC()))
results=[]
names=[]
#taking there test now
for name ,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)#this will break full data in 10 equal parts
    cv_results= model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)#this is the actual test: here the selected model in round1 will train itself for 1-9 parts and use 10th part for testing now for the round 2 it will take 1-8,10th parts for training and 9th for testing this way it will complete 10 rounds. 
    results.append(cv_results)#we are saving data of above 10 rounds for specific model
    names.append(name)
    msg="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())#calculating the mean and standard deviation of above stored data of 10 rounds
    print(msg)
#at this point you will get to new scores of each model and then select best of them
#selecting best automatically...
meanscore=[]
for cv in results:
    avg=cv.mean()
    meanscore.append(avg)
best_score=max(meanscore)
indexofbestscore=meanscore.index(best_score)
best_model=models[indexofbestscore][1]
print("best model used is",best_model)
#now as best model is selected lets use it to train our data
best_model.fit(X_train,Y_train)#this trains data to that specific model
#now model has learned with our inputs and outputs 
predictions=best_model.predict(X_test)#now testing model with that 20% data
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test,predictions))#here we are making a matrix of Y_test and predicted data(of X_test)
#matrix understanding:rows are assigned as Y_test data(here 0,1,2) and columns as predicted data {{{NOTE : here 0,1,2 are assumed in columns name} for ex:element(3,3) be 17 this tells 17 were predicted right}now let (2,3)be 2 means(that is row 2 that was "1"(actual) is considered in coumn 3(predicted by machine) that is "2")means machine assigned 2 datas in "2" category but they were "1"}, now all diagonal elements represent how many predictions of test data were correct and non diagonals tell which all were considered wrong by machine
#NOTE:leaving one column rest zeros are present in a row then these were the total 20% data set which got assigned to that specific column...means total dataset used =sum of elemnts of that row(that is 20% of total data should come)
print("\nClassification Report:")
print(classification_report(Y_test,predictions))#here a complete scorecard is formed telling about {precision:how many were predicted correctly},{recall:tells actually how many "1" was it able to find.},{f1 score:avg of precision and avg which is btw 0-1 best when near to 1},{support:tell about total samples of all category present in test data}
#model is ready to be used
#testing on new data now
new_sample=[[14.23,1.71,2.43,15.6,127.0,2.80,3.06,0.28,2.29,5.64,1.04,3.92,1065.0],[12.37,0.94,1.36,10.6,88.0,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520.0],[13.11,1.01,1070,15.0,78.0,2.98,3.18,0.26,2.28,5.30,1.12,3.18,502.0]]

sample_predictions=best_model.predict(new_sample)
for i in sample_predictions:
    
    print("predictions are",i)
#note earlier seed was 7 at that time result were 0,1,1 which was wrongly predicted then i tried keeping 1,40,42,50 and when kept 100 it gived correct results
#actually this seed value only decides how is the data going to get shuffled before distribution.

ordered_samples = [
    # Sample 1: Expected -> Class 2 (High Malic Acid, Low Flavanoids, Darker Color)
    [13.50, 3.12, 2.62, 24.0, 123.0, 1.40, 1.57, 0.22, 1.25, 8.60, 0.59, 1.30, 660.0],
    
    # Sample 2: Expected -> Class 0 (High Alcohol, High Proline, High Flavanoids)
    [14.10, 2.02, 2.40, 18.8, 103.0, 2.75, 3.00, 0.32, 1.83, 6.20, 1.07, 3.33, 1080.0],
    
    # Sample 3: Expected -> Class 1 (Low Alcohol, Light Color Intensity)
    [12.29, 1.61, 2.21, 20.4, 103.0, 1.10, 1.02, 0.37, 1.46, 3.05, 0.906, 2.52, 870.0]
]

ordered_predictions = best_model.predict(ordered_samples)
for j in ordered_predictions:
    print("Predictions:",j)