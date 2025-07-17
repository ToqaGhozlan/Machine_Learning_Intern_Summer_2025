import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv('data_encoded.csv')
print(df.head())

df = df.dropna(subset=['booking_status--converted'])

X = df[['lead time', 'special requests', 'average price ', 'repeated']]
y = df['booking_status--converted']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = RandomForestClassifier()
classifier.fit(X_train_scaled, y_train)

pickle.dump(classifier, open('MLModel.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


