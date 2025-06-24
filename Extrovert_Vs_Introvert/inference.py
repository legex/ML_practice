import requests
import json

url = 'http://localhost:8223/predict'
data = {
    "features": [0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235, -0.03482076, -0.04340085, -0.00259226, 0.01990749, -0.01764613]
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers, timeout=10)
print(response.json())