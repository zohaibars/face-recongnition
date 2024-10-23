import requests
import json

def handShake():
    url = "http://192.168.18.80:9000/handshake/"
    response = requests.get(url, headers={'Content-Type': 'application/json'})
    responseCode = response.status_code
    print(responseCode)
    data = response.json()
    print(data)
    if responseCode == 200:
        return True
    else:
        return False

if __name__ == "__main__":
    isLive = handShake()
    print('islive', isLive)
    print("Handshake done")