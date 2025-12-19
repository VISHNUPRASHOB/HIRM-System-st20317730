# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:20:49 2025

@author: vishnuprashob
"""

from flask import Flask, jsonify, request, redirect
from flasgger import Swagger
from datetime import datetime, timedelta
from collections import defaultdict

app = Flask(__name__)
Swagger(app)

# Store request timestamps per client IP
request_logs = defaultdict(list)
RATE_LIMIT = 100  # Max requests per minute
EDGE_DEVICE_URL = "http://edge-device:5000/edge"  # Replace with your edge device URL


def is_rate_limited(client_ip):
    # Disable rate limiting for localhost
    if client_ip in ("127.0.0.1", "::1"):
        return False

    now = datetime.now()
    request_logs[client_ip] = [
        t for t in request_logs[client_ip]
        if now - t < timedelta(minutes=1)
    ]
    if len(request_logs[client_ip]) >= RATE_LIMIT:
        return True
    request_logs[client_ip].append(now)
    return False


@app.route('/api', methods=['GET'])
def get_api():
    """
    A sample GET API with rate limiting.
    ---
    tags:
      - sample
    responses:
      200:
        description: Successful response
      429:
        description: Rate limit exceeded
      302:
        description: Redirect to edge device
    """
    client_ip = request.remote_addr

    # Always return 200 for localhost
    if not is_rate_limited(client_ip):
        return jsonify({"message": "Request successful", "ip": client_ip})

    # For real clients: redirect when rate-limited
    return redirect(EDGE_DEVICE_URL, code=302)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
