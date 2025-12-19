# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:31:38 2025

@author: vishnuprashob
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/edge', methods=['GET'])
def edge_api():
    return jsonify({"message": "Request handled by edge device"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
