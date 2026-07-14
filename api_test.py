import requests

# 1. Send the GET request
response = requests.get("https://api.github.com/users/octocat")

# 2. View the HTTP status code (e.g., 200 success)
print(response.status_code)

# 3. Read the data as a Python dictionary (if JSON)
if response.status_code == 200:
    data = response.json()
    print(data['name'])
    print(data['public_repos'])
    print(data['followers'])
else:
    print(f"Request failed with status code {response.status_code}")
