#!/usr/bin/env python3
"""
Test script to verify weather functionality
"""
import requests
import json

def test_weather_api():
    """Test the weather API endpoint"""
    url = "http://localhost:5000/get_weather"
    data = {"city": "test"}
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success')}")
            if result.get('success'):
                weather = result.get('weather', {})
                print(f"Weather Data: {weather}")
            else:
                print(f"Error: {result.get('error')}")
        else:
            print("Request failed")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_weather_api() 