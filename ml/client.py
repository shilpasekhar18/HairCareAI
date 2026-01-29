import requests

url = "http://127.0.0.1:5001/predict"

data = {
    "Id": 1,
    "Age": 25,
    "Genetics": "Yes",
    "Hormonal Changes": "No",
    "Medical Conditions": "None",
    "Medications & Treatments": "No",
    "Nutritional Deficiencies ": "Iron",
    "Stress": "High",
    "Poor Hair Care Habits ": "Yes",
    "Environmental Factors": "Pollution",
    "Smoking": "No",
    "Weight Loss ": "No"
}

response = requests.post(url, json=data)
print(response.json())
