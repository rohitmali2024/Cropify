from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import numpy as np
import pandas
import sklearn
import pickle
import json
import os
import requests
from datetime import datetime
import platform
import psutil
import time
import random

# Load models with proper path handling for Vercel
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'model.pkl')
sc_path = os.path.join(current_dir, '..', 'standscaler.pkl')
mx_path = os.path.join(current_dir, '..', 'minmaxscaler.pkl')

model = pickle.load(open(model_path,'rb'))
sc = pickle.load(open(sc_path,'rb'))
mx = pickle.load(open(mx_path,'rb'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Weather API Configuration
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', "YOUR_OPENWEATHERMAP_API_KEY")
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Language translations - Indian Languages Only
TRANSLATIONS = {
    'en': {
        'title': 'Smart Crop Recommendation System',
        'subtitle': 'Get AI-powered recommendations for the best crops to cultivate based on your soil and climate conditions',
        'navbar_brand': 'Smart Crop Recommendation',
        'soil_nutrients': 'Soil Nutrients Analysis',
        'soil_nutrients_desc': 'Advanced analysis of Nitrogen, Phosphorus, and Potassium levels for optimal crop selection',
        'climate_intelligence': 'Climate Intelligence',
        'climate_intelligence_desc': 'Comprehensive evaluation of Temperature, Humidity, and Rainfall patterns',
        'ai_insights': 'AI-Powered Insights',
        'ai_insights_desc': 'Machine Learning algorithms provide precise crop recommendations with high accuracy',
        'nitrogen': 'Nitrogen (N)',
        'phosphorus': 'Phosphorus (P)',
        'potassium': 'Potassium (K)',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph_level': 'pH Level',
        'rainfall': 'Rainfall',
        'nitrogen_placeholder': 'Enter Nitrogen content',
        'phosphorus_placeholder': 'Enter Phosphorus content',
        'potassium_placeholder': 'Enter Potassium content',
        'temperature_placeholder': 'Enter Temperature in °C',
        'humidity_placeholder': 'Enter Humidity in %',
        'ph_placeholder': 'Enter pH value (0-14)',
        'rainfall_placeholder': 'Enter Rainfall in mm',
        'get_recommendation': 'Get Smart Recommendation',
        'loading_text': 'Analyzing your data and generating recommendations...',
        'recommended_crop': '🌾 Recommended Crop',
        'crop_recommendation': '{} is the best crop to be cultivated right there',
        'error_message': 'Sorry, we could not determine the best crop to be cultivated with the provided data.',
        'language': 'Language',
        'english': 'English',
        'hindi': 'Hindi',
        'bengali': 'Bengali',
        'telugu': 'Telugu',
        'marathi': 'Marathi',
        'tamil': 'Tamil',
        'gujarati': 'Gujarati',
        'kannada': 'Kannada',
        'malayalam': 'Malayalam',
        'punjabi': 'Punjabi',
        'odia': 'Odia',
        'assamese': 'Assamese',
        'sanskrit': 'Sanskrit',
        'weather_data': 'Weather Data',
        'weather_data_desc': 'Get real-time weather data from your location or enter manually',
        'auto_weather': 'Auto Weather',
        'manual_weather': 'Manual Weather',
        'get_location': 'Get My Location',
        'location_placeholder': 'Enter city name',
        'current_weather': 'Current Weather',
        'weather_temp': 'Temperature',
        'weather_humidity': 'Humidity',
        'weather_rainfall': 'Rainfall',
        'weather_loading': 'Fetching weather data...',
        'weather_error': 'Unable to fetch weather data. Please try again or enter manually.',
        'location_error': 'Unable to get your location. Please enter city name manually.',
        'use_weather_data': 'Use Weather Data',
        'enter_manually': 'Enter Manually',
        'weather_success': 'Weather data fetched successfully!',
        'location_permission': 'Please allow location access to get weather data automatically.'
    }
    # Note: Truncated other languages for brevity - you can add them back if needed
}

def get_weather_data(city):
    """Fetch weather data from OpenWeatherMap API"""
    try:
        if WEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
            return None
        
        url = f"{WEATHER_BASE_URL}?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': round(data['main']['temp'], 1),
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0),
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon']
            }
        return None
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def get_weather_by_coordinates(lat, lon):
    """Fetch weather data using coordinates"""
    try:
        if WEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
            return None
        
        url = f"{WEATHER_BASE_URL}?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': round(data['main']['temp'], 1),
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0),
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'city': data['name']
            }
        return None
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

@app.route('/')
def home():
    language = session.get('language', 'en')
    translations = TRANSLATIONS.get(language, TRANSLATIONS['en'])
    return render_template('index.html', **translations)

@app.route('/get_weather', methods=['POST'])
def get_weather():
    try:
        data = request.get_json()
        city = data.get('city', '').strip()
        
        if not city:
            return jsonify({'success': False, 'error': 'City name is required'})
        
        weather_data = get_weather_data(city)
        
        if weather_data:
            return jsonify({'success': True, 'data': weather_data})
        else:
            return jsonify({'success': False, 'error': 'Unable to fetch weather data'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_weather_by_location', methods=['POST'])
def get_weather_by_location():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if not lat or not lon:
            return jsonify({'success': False, 'error': 'Coordinates are required'})
        
        weather_data = get_weather_by_coordinates(lat, lon)
        
        if weather_data:
            return jsonify({'success': True, 'data': weather_data})
        else:
            return jsonify({'success': False, 'error': 'Unable to fetch weather data'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            language = session.get('language', 'en')
            translations = TRANSLATIONS.get(language, TRANSLATIONS['en'])
            result = translations['crop_recommendation'].format(crop)
        else:
            language = session.get('language', 'en')
            translations = TRANSLATIONS.get(language, TRANSLATIONS['en'])
            result = translations['error_message']
        
        return render_template('index.html', prediction=result, **translations)
    
    except Exception as e:
        language = session.get('language', 'en')
        translations = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        error_msg = translations['error_message']
        return render_template('index.html', prediction=error_msg, **translations)

@app.route('/change_language/<language>')
def change_language(language):
    if language in TRANSLATIONS:
        session['language'] = language
    return redirect(url_for('home'))

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda status, headers: None)

# For Vercel deployment
app.template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
app.static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

if __name__ == '__main__':
    app.run(debug=True)
