# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:30:39 2025

@author: vishnuprashob
"""

import requests
import threading
import time

API_URL = "http://127.0.0.1:5000/api"

def call_api():
    while True:
        try:
            response = requests.get(API_URL)
            print(f"Response: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(0.1)  # Adjust delay to control request rate

# Simulate 150 requests (exceeds the 100/minute limit)
threads = []
for _ in range(150):
    t = threading.Thread(target=call_api)
    threads.append(t)
    t.start()

for t in threads:
    t.join()
