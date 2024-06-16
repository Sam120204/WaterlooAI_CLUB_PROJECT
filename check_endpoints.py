import requests

base_url = "http://localhost:8000"

endpoints = [
    "/", 
    "/docs", 
    "/openapi.json", 
    "/api/v1/tenants", 
    "/api/v1/collections"
]

for endpoint in endpoints:
    try:
        response = requests.get(base_url + endpoint)
        print(f"Endpoint: {endpoint}, Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error accessing endpoint {endpoint}: {e}")
