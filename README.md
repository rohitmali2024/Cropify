# Smart Crop Recommendation System

A modern, AI-powered crop recommendation system that helps farmers choose the best crops to cultivate based on soil nutrients and climate conditions. The system features real-time weather data integration, multilingual support for Indian languages, and a beautiful, responsive user interface.

## 🌟 Features

### 🤖 AI-Powered Recommendations
- Machine Learning model trained on comprehensive agricultural data
- Analyzes soil nutrients (Nitrogen, Phosphorus, Potassium)
- Evaluates climate conditions (Temperature, Humidity, pH, Rainfall)
- Provides accurate crop recommendations with high precision

### 🌤️ Real-Time Weather Integration
- **Auto Weather Mode**: Automatically fetch weather data from user's location
- **Manual Weather Mode**: Traditional manual data entry
- **Location Detection**: Uses browser geolocation for automatic weather retrieval
- **City Search**: Enter any city name to get local weather data
- **Weather Display**: Shows current temperature, humidity, and rainfall with weather icons

### 🌍 Multilingual Support
Supports 13 Indian languages:
- English, Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Sanskrit

### 🎨 Modern User Interface
- Beautiful gradient animations and glassmorphism effects
- Responsive design for all devices
- Interactive weather cards and smooth animations
- Professional color scheme and typography

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Crop_Recommendation-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weather API (Optional)**
   - Get a free API key from [OpenWeatherMap](https://openweathermap.org/)
   - Update `WEATHER_API_KEY` in `app.py`
   - See `WEATHER_SETUP.md` for detailed instructions

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:5000`

## 📊 How It Works

### 1. Data Input
Users can provide soil and climate data in two ways:
- **Manual Entry**: Traditional form-based input
- **Auto Weather**: Real-time weather data from their location

### 2. AI Analysis
The system uses a trained Random Forest model to analyze:
- **Soil Nutrients**: N, P, K levels for optimal crop selection
- **Climate Factors**: Temperature, humidity, pH, and rainfall patterns

### 3. Smart Recommendations
Based on the analysis, the system recommends the best crop from 22 different options including:
- Grains: Rice, Maize
- Fruits: Apple, Mango, Banana, Orange, Grapes
- Vegetables: Lentil, Chickpea, Kidney beans
- Cash Crops: Cotton, Jute, Coffee
- And many more...

## 🛠️ Technical Stack

### Backend
- **Flask**: Web framework for the application
- **scikit-learn**: Machine Learning model (Random Forest)
- **numpy & pandas**: Data processing and analysis
- **requests**: Weather API integration

### Frontend
- **HTML5/CSS3**: Modern, responsive design
- **JavaScript**: Interactive features and weather integration
- **Bootstrap 5.3**: UI framework
- **FontAwesome**: Icons and visual elements

### Machine Learning
- **Model**: Random Forest Classifier
- **Preprocessing**: MinMaxScaler and StandardScaler
- **Accuracy**: High precision crop recommendations

## 📁 Project Structure

```
Crop_Recommendation-main/
├── app.py                          # Main Flask application
├── model.pkl                       # Trained ML model
├── minmaxscaler.pkl               # MinMaxScaler for preprocessing
├── standscaler.pkl                # StandardScaler for preprocessing
├── Crop_recommendation.csv        # Training dataset
├── requirements.txt               # Python dependencies
├── WEATHER_SETUP.md              # Weather API setup guide
├── templates/
│   └── index.html                # Main HTML template
└── static/
    └── crop.png                  # Application icon
```

## 🌐 API Endpoints

- `GET /`: Main application page
- `POST /get_weather`: Fetch weather data by city name
- `POST /get_weather_by_location`: Fetch weather data by coordinates
- `POST /predict`: Get crop recommendation
- `GET /change_language/<language>`: Switch application language

## 🎯 Use Cases

### For Farmers
- Get data-driven crop recommendations
- Access real-time weather information
- Plan farming activities based on current conditions
- Optimize resource allocation

### For Agricultural Consultants
- Provide scientific recommendations to clients
- Use as a reference tool for crop planning
- Demonstrate modern agricultural technology

### For Students/Researchers
- Study machine learning in agriculture
- Understand crop-climate relationships
- Learn about modern farming technology

## 🔧 Configuration

### Weather API Setup
1. Sign up at [OpenWeatherMap](https://openweathermap.org/)
2. Get your API key
3. Update `WEATHER_API_KEY` in `app.py`
4. Restart the application

### Language Configuration
- The system automatically detects user's language preference
- Language selection persists across sessions
- All UI elements are translated to selected language

## 🚨 Important Notes

### Security
- Never commit API keys to version control
- Use environment variables in production
- Keep dependencies updated

### Weather API Limits
- Free tier: 1000 calls/day
- Consider upgrading for commercial use
- Monitor API usage to avoid rate limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section in `WEATHER_SETUP.md`
2. Review the application logs
3. Ensure all dependencies are installed correctly

---

**Built with ❤️ for the agricultural community**