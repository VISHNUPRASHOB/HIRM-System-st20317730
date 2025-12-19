# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:00:51 2025

@author: vishnuprashob
"""

from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(0.1, 1)  # Random wait between requests

    @task
    def call_api(self):
        self.client.get("/api")