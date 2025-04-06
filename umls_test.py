import requests

base_url = "https://uts-ws.nlm.nih.gov/rest"
api_key = "a9383cc4-f6e1-442f-bb44-c44935d4f8d8"
cui = "C0009044"  # Example CUI

# Construct the request URL with the API key
request_url = f"{base_url}/content/current/CUI/{cui}?apiKey={api_key}"

response = requests.get(request_url)
response.raise_for_status()

data = response.json()
print(data)