import requests

url = 'http://localhost:5000/predict_api'

data = {
    'radius_mean': 17.99, 
    'texture_mean': 10.38, 
    'perimeter_mean': 122.80, 
    'area_mean': 1001.0
}
r = requests.post(url, json=data)
print(r.json())