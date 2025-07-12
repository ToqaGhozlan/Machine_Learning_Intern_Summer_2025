import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as y1
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

wr.filterwarnings('ignore')


data_file = pd.read_csv("C:\\Users\\Al Badr\\Desktop\\Yousef Mahmoud Ali_ML_GP2\\first inten project.csv")
data_file.columns = [col.strip() for col in data_file.columns]
outliers1 = data_file.select_dtypes(include=['int64', 'float64']).columns.tolist();
print("Numerical columns: " , outliers1)

before_data = data_file.copy();

def IQR_METHOD(data , col):
    Q1 = data[col].quantile(0.25);
    Q3 = data[col].quantile(0.75);
    IQR = Q3 - Q1;
    lower_bound = Q1 - (1.5 * IQR);
    upper_bound = Q3 + (1.5 * IQR);
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
print("\n")
for col in outliers1: 
    before_removing = data_file.shape[0];
    data_file = IQR_METHOD(data_file, col)
    after_removing = data_file.shape[0];
    print(f"\n{col}: removed {before_removing - after_removing} outliers using IQR Method")


y1.figure(figsize=(10,5))
y1.title("Before")
y1.boxplot(before_data[outliers1].values, labels=outliers1, vert=False)
plt.show();

y1.figure(figsize=(10,5))
y1.title("After")
y1.boxplot(data_file[outliers1].values ,labels = outliers1, vert=False)
y1.show()

    

print("Shape: " , data_file.shape)
print("First 5 rows: "); print(data_file.head());
print("Data types: "); print(data_file.dtypes);
print("Nulls:\n" , data_file.isnull().sum())

plt.figure(figsize=(10,5))
sns.heatmap(data_file.select_dtypes(include=['int64','float64']).corr(), annot=True, cmap = "coolwarm", fmt=".2f")
plt.title("Feature Correlation HeatMap (After)")
plt.show()
plt.figure(figsize=(10,5))
sns.heatmap(before_data.select_dtypes(include=['int64','float64']).corr(), annot=True, cmap = "coolwarm", fmt=".2f")
plt.title("Feature Correlation HeatMap (Before)")
plt.show()

data_file = data_file.drop(columns=['repeated', 'P-not-C' , 'car parking space' , 'number of week nights'])

cat_col = data_file.select_dtypes(include=['object', 'category']).columns.tolist();
print("Categorical columns: " , cat_col);
for col in cat_col:
    data_file[col] = data_file[col].str.strip()

data_file['date of reservation'] = pd.to_datetime(data_file['date of reservation'], errors='coerce')
data_file['reservation_month'] = data_file['date of reservation'].dt.to_period("M")
data_file['reservation_day'] = data_file['date of reservation'].dt.day
data_file['reservation_weekday'] = data_file['date of reservation'].dt.dayofweek

data_file = data_file.dropna()
encode = LabelEncoder()
data_file = pd.get_dummies(data_file , columns=['type of meal', 'room type', 'market segment type'], drop_first=True)#One-Hot
data_file['booking status'] = encode.fit_transform(data_file['booking status'])#label encoding


X = data_file.drop(columns=['booking status','Booking_ID', 'date of reservation' , 'reservation_month'])
Y = data_file['booking status']

X_train, X_test, Y_train, Y_test = train_test_split(X , Y , test_size=0.2, random_state=30)
print("Training set: " , X_train.shape)
print("Test split: " , X_test.shape)
Model = LogisticRegression(max_iter=2000)
Model.fit(X_train , Y_train)
y_predict = Model.predict(X_test)
Acc = accuracy_score(Y_test , y_predict)
print("Accuracy: " , round(Acc * 100, 2), "%")
print("Classifications report:",classification_report(Y_test , y_predict))
print("Confusion matrix:",confusion_matrix(Y_test , y_predict))


ConfusionMatrixDisplay.from_estimator(Model, X_test, Y_test)
plt.title("Confusion Matrix")
plt.show()