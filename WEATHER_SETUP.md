# Weather API Setup Guide

## Overview
This crop recommendation system now includes real-time weather data integration using the OpenWeatherMap API. Users can either manually enter weather data or automatically retrieve it from their location.

## Setup Instructions

### 1. Get OpenWeatherMap API Key
1. Go to [OpenWeatherMap](https://openweathermap.org/)
2. Sign up for a free account
3. Navigate to "My API Keys" section
4. Copy your API key

### 2. Configure the API Key
1. Open `app.py`
2. Find the line: `WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"`
3. Replace `"YOUR_OPENWEATHERMAP_API_KEY"` with your actual API key
4. Save the file

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

## Features

### Auto Weather Mode
- **Get My Location**: Automatically detects user's location and fetches weather data
- **City Search**: Enter a city name to get weather data for that location
- **Weather Display**: Shows current temperature, humidity, and rainfall
- **Use Weather Data**: Automatically populates the form with fetched weather data

### Manual Weather Mode
- Users can manually enter all weather parameters
- Traditional form-based input

## Weather Data Sources
- **Temperature**: Current temperature in Celsius
- **Humidity**: Current humidity percentage
- **Rainfall**: Current rainfall in mm (if available)

## API Endpoints
- `GET /`: Main application page
- `POST /get_weather`: Fetch weather by city name
- `POST /get_weather_by_location`: Fetch weather by coordinates
- `POST /predict`: Get crop recommendation

## Error Handling
- Location permission denied
- Invalid city name
- API service unavailable
- Network connectivity issues

## Security Notes
- Keep your API key secure and never commit it to version control
- Consider using environment variables for production deployment
- The API key is free for limited usage (1000 calls/day)

## Troubleshooting
1. **API Key Error**: Ensure your API key is correct and active
2. **Location Not Working**: Check browser permissions for location access
3. **City Not Found**: Verify city name spelling and try alternative names
4. **Network Issues**: Check internet connectivity and API service status 