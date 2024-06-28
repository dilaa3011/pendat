import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

df_data = pd.read_excel('balloons.xlsx')
df = df_data

from sklearn.preprocessing import LabelEncoder
df = df_data
label_encoders = {}
for column in df_data.columns:
    if df_data[column].dtype == 'object' and column != 'inflated':  # Kolom 'inflated' tidak diubah
        le = LabelEncoder()
        df_data[column] = le.fit_transform(df_data[column])
        label_encoders[column] = le
print(df)

X = df_data[['color', 'size', 'act', 'age']]
y = df_data['inflated']

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
model_rfc = RandomForestClassifier()
model_rfc.fit(X_train, y_train)

# Memprediksi pada data uji
y_pred = model_rfc.predict(X_test)

# Evaluasi model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
with open('model.pkl','wb')as file:
    pickle.dump(model_rfc,file)