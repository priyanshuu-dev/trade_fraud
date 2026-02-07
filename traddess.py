import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("trades.csv")

df_encoded=pd.get_dummies(df,columns=["paymentMethod"],dtype=int)
print(df_encoded.dtypes)

x=df_encoded.drop(columns="label")
y=df_encoded["label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

model=XGBClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(df.dtypes)

user={
    "accountAgeDays":int(input("Enter AccountDays: ")),
    "numItems":int(input("Enter NumItems: ")),
    "localTime":float(input("Enter LocalTime: ")),
    "paymentMethod":input("Enter Payment method: "),
    "paymentMethodAgeDays":float(input("Enter PaymentMethodAgeDays: "))
    
    }

user_input_dataframe=pd.DataFrame([user])
user_input_dataframe=pd.get_dummies(user_input_dataframe)
user_input_ready=user_input_dataframe.reindex(columns=x.columns,fill_value=0)
label_output=model.predict(user_input_ready)
if label_output==0:
    print("No Faud Detected")
else:
    print("Fraud Detected")

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score: ",accuracy)
