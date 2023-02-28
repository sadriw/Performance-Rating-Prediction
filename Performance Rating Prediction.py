#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[4]:


# load data
data = pd.read_csv('HR-Employee-Attrition.csv')
data.head()


# In[8]:


#cek data kosong
data.info()


# In[9]:


# melakukan label encoding pada kolom 
le = LabelEncoder()
data['BusinessTravel_encoded'] = le.fit_transform(data['BusinessTravel'])
data['Attrition_encoded'] = le.fit_transform(data['Attrition'])
data['Department_encoded'] = le.fit_transform(data['Department'])
data['EducationField_encoded'] = le.fit_transform(data['EducationField'])
data['Gender_encoded'] = le.fit_transform(data['Gender'])
data['JobRole_encoded'] = le.fit_transform(data['JobRole'])
data['MaritalStatus_encoded'] = le.fit_transform(data['MaritalStatus'])
data['Over18_encoded'] = le.fit_transform(data['Over18'])
data['OverTime_encoded'] = le.fit_transform(data['OverTime'])
print(data)


# In[10]:


data.drop(['BusinessTravel'], axis=1, inplace=True)
data.drop(['Attrition'], axis=1, inplace=True)
data.drop(['EducationField'], axis=1, inplace=True)
data.drop(['Gender'], axis=1, inplace=True)
data.drop(['JobRole'], axis=1, inplace=True)
data.drop(['MaritalStatus'], axis=1, inplace=True)
data.drop(['Over18'], axis=1, inplace=True)
data.drop(['OverTime'], axis=1, inplace=True)
data.drop(['Department'], axis=1, inplace=True)


# In[12]:


data.info()


# In[16]:


# ubah tipe data kolom "nilai" menjadi float64
data['BusinessTravel_encoded'] = pd.to_numeric(data['BusinessTravel_encoded'], errors='coerce')
data['Attrition_encoded'] = pd.to_numeric(data['Attrition_encoded'], errors='coerce')
data['Department_encoded '] = pd.to_numeric(data['Department_encoded'], errors='coerce')
data['EducationField_encoded'] = pd.to_numeric(data['EducationField_encoded'], errors='coerce')
data['Gender_encoded'] = pd.to_numeric(data['Gender_encoded'], errors='coerce')
data['JobRole_encoded'] = pd.to_numeric(data['JobRole_encoded'], errors='coerce')
data['MaritalStatus_encoded'] = pd.to_numeric(data['MaritalStatus_encoded'], errors='coerce')
data['Over18_encoded'] = pd.to_numeric(data['Over18_encoded'], errors='coerce')
data['OverTime_encoded'] = pd.to_numeric(data['OverTime_encoded'], errors='coerce')


# In[18]:


# memisahkan data ke dalam fitur dan target
X = data.drop('PerformanceRating', axis=1)
y = data['PerformanceRating']


# In[19]:


# membagi data ke dalam data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


# membuat model Naive Bayes
model = GaussianNB()


# In[21]:


# melatih model dengan data latih
model.fit(X_train, y_train)


# In[22]:


# melakukan prediksi dengan data uji
y_pred = model.predict(X_test)


# In[23]:


# menghitung akurasi prediksi
accuracy = accuracy_score(y_test, y_pred)


# In[24]:


# menampilkan hasil akurasi prediksi
print("Akurasi prediksi: {:.2f}%".format(accuracy*100))


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix


# In[39]:


# menghitung metrik koefisien
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))


# In[40]:


# evaluasi model
if accuracy >= 0.8:
    print("\nModel memiliki performa yang baik")
else:
    print("\nModel memiliki performa yang buruk")


# In[ ]:




