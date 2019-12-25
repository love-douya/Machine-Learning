import re
import requests

response = requests.get("http://python.org")
html = response.text
print(len(html))

tokens = [tok for tok in html.split()]

