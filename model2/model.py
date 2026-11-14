import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
#NOTE:CLASSIFIERS:upper wale Yeh kab use hota hai: Jab tumhara output koi Category ya Group ho (jaise: Yes / No, Spam / Not Spam, Cat / Dog, ya Iris flower ki species(as done earlier))
#NOTE:here we use Regression:Yeh kab use hota hai: Jab tumhara output ek Continuous Number (counting ya price) ho (jaise: ghar ki price, kal ka temperature, ya Bitcoin ka price).
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#laoding data
import yfinance as yf
#extracting Bitcoin 5 years of data
btc_data=yf.download("BTC-USD",period="5y",interval="1d")
print(btc_data.head())#checking first 5 rows
#saving in csv file
#btc_data.to_csv(r"C:\Users\Raj\OneDrive\Documents\bitcoin_data.csv") #saved it ones
#df=pd.read_csv(r"C:\Users\Raj\OneDrive\Documents\bitcoin_data.csv",skiprows=[1,2]) jab windows mein run kiya
df=pd.read_csv("/mnt/c/Users/Raj/OneDrive/Documents/bitcoin_data.csv")#now because of issue we are running in linux so this type of locations are present
#now how will it predict:we will actually give todays data for high,low,open,volume and get prediction for tommorows data for close. so hame close wale column ka data ek row upper shift karna hoga
df['Target_next_close']=df['Close'].shift(-1)#isse new column banega and usska data Close wale column ka hoga jo ki ek row upper shifted hoga NOTE: isse ek new row banjayegi nan value se filled hoga date wali row ke ange
New=df.drop(columns=['Close']).dropna()#taaki nan wale rows hat jaye jo ki last wali hai because hamne ek row upper shift kiya tha 
print(New)
array=New.values
X=array[2:,1:5]#row 2 se data tha and 0to 4 th column tak input dena tha toh 5 tak range denge because last is not considered## also date column sholud not be considered
Y=array[2:,5]#last column only needed
#NOTE:csv se imported data mein two rows useless thi unko hata 1st and 2nd used skip rows
#NOTE:yaha ek toh shift kiya toh jo last column hai ussmein hi nan agaya so jab model testing mein maths use hui error agaya so dropna se hataya
#NOTE:test data mein date wala column bhi dediya jo ki accuracy and maths ca;ulation ka part nahi tha toh non float datatype error aaya
#lets check best model to be used
validation_size=0.2
seed=60
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
models=[]
# models.append(("LR",LogisticRegression()))
# models.append(("LDA",LinearDiscriminantAnalysis()))
# models.append(("KNN",KNeighborsClassifier()))
# models.append(("CART",DecisionTreeClassifier()))
# models.append(("NB",GaussianNB()))
# models.append(("SVM",SVC()))
models.append(("LR", LinearRegression()))
models.append(("KNN", KNeighborsRegressor()))
models.append(("CART", DecisionTreeRegressor()))
models.append(("SVR", SVR()))
results=[]
names=[]
scoring="r2"#yaha accuracy(yeh sirf sahi/galat per kaam karta hai) ki jagah R-squared score yeh jitna 1 ke kareeb hoga utna best model hoga(yeh kitna sahi hai ussper kaam karta hai)
for name ,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed,shuffle=True) #yeh again 10 parts mein total data ko break karega and har ek model 9 parts train 10th per test then 1to 8,10 per train 9th per test this way each model does 10 times and all observations for each is stored in a list for all then rounds
    cv_results= model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring) 
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)
# from sklearn.metrics import accuracy_score, classification_report,confusion_matrix  ## issmein yeh sab nahi hota yeh bas yes/no type mein hi use karte hain
meanscore=[]
for cv in results:#result ke ander har ek ki observations stored hain in form of arrays
    avg=cv.mean()
    meanscore.append(avg)
best_score=max(meanscore)
indexofbestscore=meanscore.index(best_score)
best_model=models[indexofbestscore][1]
print("best model used is",best_model)
best_model.fit(X_train,Y_train)
#now that 20% data can be tested 
predictions=best_model.predict(X_test)#yeh hai un 20% data ka prediction now we will see how much is it correct by comparing its r2 values
#real was Y_test and machines prediction are "predictions" now lets compare
Y_test = pd.to_numeric(Y_test, errors='coerce')#string ko numeric banata hai and coerce agar faltu string like hello ho then instead of error gives it nan value
comparision=pd.DataFrame({"REAL_PRICE":Y_test,"PREDICTED_PRICE":predictions,"Difference":Y_test-predictions})
print(comparision)
#now checking on real data 
#we are extracting real time todays data

# 1. Abhi ka fresh Bitcoin data download karo (pichle 5 din ka)
live_df = yf.download("BTC-USD", period="5d", interval="1d", multi_level_index=False)

# 2.last row (Aaj/Latest din) ke High, Low, Open, Volume nikalo
# NOTE: order wahi hona chahiye jo training mein tha -> ['High', 'Low', 'Open', 'Volume']
latest_row = live_df[['High', 'Low', 'Open', 'Volume']].iloc[[-2]].values.astype(float)#.iloc[[-1]] dataframe ka last row uthata hai and astype saari values ko float mein convert kardeta hai

np.set_printoptions(suppress=True,precision=2)#supress =True e+04 types off kardega and precision =2 will present only 2 digits after
pd.set_option('display.float_format','{:.2f}'.format)
# 3. Apne trained model se kal ka price predict karwate hain
tomorrows_price = best_model.predict(latest_row)
print("Aaj ka data jo model ko diya gaya:", latest_row)
# print(f"🚀 Kal ke liye Predicted Close Price: ${tomorrows_price[0]:.2f}")#yeh data exponential form mein aata hai like  $7.34e+04=$7.34*10^4 hai ok

print(f"🚀 Kal ke liye Predicted Close Price: ${tomorrows_price[0]:.2f}")
print(live_df)
#now we can save this model so that we dont have to train it again and again and then upload that saved model to a website so that it can be accessed very easily
import joblib
joblib.dump(best_model, "bitcoin_model.pkl")#isse hamara model(best_model) is bitcoin_model.pkl mein save hojayega