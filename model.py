import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

df=pd.read_csv('C:/Users/svija/OneDrive/Desktop/ckd prediction/dataset/kidney_disease.csv')
categorical=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']
numerical=['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']

del df['id']

#filling missing values with mode
cols=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane','classification']
for i in cols:
    df[i]=df[i].fillna(df[i].dropna().mode()[0])
df['dm']=df['dm'].str.lstrip(' ')
#changing categorical variables into numerical variables
cleanup_nums = {"rbc":    {"normal": 0, "abnormal": 1},
                "pc":     {"normal": 0, "abnormal": 1},
                "pcc":    {'present':0, 'notpresent':1},
                "ba":     {'present':0, 'notpresent':1},
                "htn":    {"yes": 0, "no": 1},
                "dm":     {"yes": 0, "no": 1},
                "cad":    {"yes": 0, "no": 1},
                "appet":  {"good":0, "poor":1},
                "pe":     {"yes": 0, "no": 1},
                "ane":    {"yes": 0, "no": 1},
                "classification":{"ckd": 0, "notckd": 1}}
df.replace(cleanup_nums, inplace=True)
df.dtypes

train=df.iloc[:,:-1]
test=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(train.values,test.values,test_size=0.2,random_state=0)

RF=RandomForestClassifier()
#Train the model
RF.fit(X_train,y_train)
# Saving model to disk
pickle.dump(RF, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


print(model.predict([[ 58,80, 1.025,0,0,0,0,1,1,131,18,1.1,141,3.5,15.8,53,6800,6.1,1,1,1,0,1,1]]))#negative
print(model.predict([[ 75,80,1.02,2,0,1,1,1,1,253,142,4.6,138,5.8,16.5,33,7200,4.3,0,0,0,0,1,1]]))#positive
