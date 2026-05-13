#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary module
import pickle

import pandas as pd

# Load model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Input
pregnancies(count) = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
bp = float(input("Blood Pressure: "))
skin thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# DataFrame
input_data = pd.DataFrame([{
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': bp,
    'SkinThickness': skin,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}])

# Scale input
input_scaled = scaler.transform(input_data)
input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)

# Prediction
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

#DIABETES PREDICTION SYSTEM
print("\n DIABETES PREDICTION SYSTEM")
print("-"*40)

if prediction[0] == 1:
    print("Result: DIABETIC")
else:
    print("Result: NOT DIABETIC")

print(f"Probability: {probability:.2%}")


# In[ ]:




