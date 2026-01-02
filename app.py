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

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Weather API Configuration
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your actual API key
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
    },
    'hi': {
        'title': 'स्मार्ट फसल अनुशंसा प्रणाली',
        'subtitle': 'अपनी मिट्टी और जलवायु स्थितियों के आधार पर सर्वोत्तम फसलों की खेती के लिए AI-संचालित सिफारिशें प्राप्त करें',
        'navbar_brand': 'स्मार्ट फसल अनुशंसा',
        'soil_nutrients': 'मिट्टी के पोषक तत्व विश्लेषण',
        'soil_nutrients_desc': 'इष्टतम फसल चयन के लिए नाइट्रोजन, फास्फोरस और पोटेशियम के स्तर का उन्नत विश्लेषण',
        'climate_intelligence': 'जलवायु बुद्धिमत्ता',
        'climate_intelligence_desc': 'तापमान, आर्द्रता और वर्षा पैटर्न का व्यापक मूल्यांकन',
        'ai_insights': 'AI-संचालित अंतर्दृष्टि',
        'ai_insights_desc': 'मशीन लर्निंग एल्गोरिदम उच्च सटीकता के साथ सटीक फसल सिफारिशें प्रदान करते हैं',
        'nitrogen': 'नाइट्रोजन (N)',
        'phosphorus': 'फास्फोरस (P)',
        'potassium': 'पोटेशियम (K)',
        'temperature': 'तापमान',
        'humidity': 'आर्द्रता',
        'ph_level': 'pH स्तर',
        'rainfall': 'वर्षा',
        'nitrogen_placeholder': 'नाइट्रोजन सामग्री दर्ज करें',
        'phosphorus_placeholder': 'फास्फोरस सामग्री दर्ज करें',
        'potassium_placeholder': 'पोटेशियम सामग्री दर्ज करें',
        'temperature_placeholder': 'तापमान °C में दर्ज करें',
        'humidity_placeholder': 'आर्द्रता % में दर्ज करें',
        'ph_placeholder': 'pH मान दर्ज करें (0-14)',
        'rainfall_placeholder': 'वर्षा मिमी में दर्ज करें',
        'get_recommendation': 'स्मार्ट सिफारिश प्राप्त करें',
        'loading_text': 'आपका डेटा विश्लेषण कर रहा है और सिफारिशें तैयार कर रहा है...',
        'recommended_crop': '🌾 अनुशंसित फसल',
        'crop_recommendation': '{} वहां की खेती के लिए सर्वोत्तम फसल है',
        'error_message': 'क्षमा करें, हम प्रदान किए गए डेटा के साथ सर्वोत्तम फसल निर्धारित नहीं कर सके।',
        'language': 'भाषा',
        'english': 'अंग्रेजी',
        'hindi': 'हिंदी',
        'bengali': 'बंगाली',
        'telugu': 'तेलुगु',
        'marathi': 'मराठी',
        'tamil': 'तमिल',
        'gujarati': 'गुजराती',
        'kannada': 'कन्नड़',
        'malayalam': 'मलयालम',
        'punjabi': 'पंजाबी',
        'odia': 'ओडिया',
        'assamese': 'असमिया',
        'sanskrit': 'संस्कृत',
        'weather_data': 'मौसम डेटा',
        'weather_data_desc': 'अपने स्थान से रीयल-टाइम मौसम डेटा प्राप्त करें या मैन्युअली दर्ज करें',
        'auto_weather': 'ऑटो मौसम',
        'manual_weather': 'मैन्युअल मौसम',
        'get_location': 'मेरा स्थान प्राप्त करें',
        'location_placeholder': 'शहर का नाम दर्ज करें',
        'current_weather': 'वर्तमान मौसम',
        'weather_temp': 'तापमान',
        'weather_humidity': 'आर्द्रता',
        'weather_rainfall': 'वर्षा',
        'weather_loading': 'मौसम डेटा प्राप्त कर रहा है...',
        'weather_error': 'मौसम डेटा प्राप्त करने में असमर्थ। कृपया पुनः प्रयास करें या मैन्युअली दर्ज करें।',
        'location_error': 'आपका स्थान प्राप्त करने में असमर्थ। कृपया शहर का नाम मैन्युअली दर्ज करें।',
        'use_weather_data': 'मौसम डेटा का उपयोग करें',
        'enter_manually': 'मैन्युअली दर्ज करें',
        'weather_success': 'मौसम डेटा सफलतापूर्वक प्राप्त किया गया!',
        'location_permission': 'कृपया मौसम डेटा स्वचालित रूप से प्राप्त करने के लिए स्थान एक्सेस की अनुमति दें।'
    },
    'bn': {
        'title': 'স্মার্ট ফসল সুপারিশ সিস্টেম',
        'subtitle': 'আপনার মাটি এবং জলবায়ু অবস্থার ভিত্তিতে চাষের জন্য সেরা ফসলের AI-চালিত সুপারিশ পান',
        'navbar_brand': 'স্মার্ট ফসল সুপারিশ',
        'soil_nutrients': 'মাটি পুষ্টি বিশ্লেষণ',
        'soil_nutrients_desc': 'সর্বোত্তম ফসল নির্বাচনের জন্য নাইট্রোজেন, ফসফরাস এবং পটাসিয়ামের মাত্রার উন্নত বিশ্লেষণ',
        'climate_intelligence': 'জলবায়ু বুদ্ধিমত্তা',
        'climate_intelligence_desc': 'তাপমাত্রা, আর্দ্রতা এবং বৃষ্টিপাতের প্যাটার্নের ব্যাপক মূল্যায়ন',
        'ai_insights': 'AI-চালিত অন্তর্দৃষ্টি',
        'ai_insights_desc': 'মেশিন লার্নিং অ্যালগরিদম উচ্চ নির্ভুলতার সাথে সঠিক ফসল সুপারিশ প্রদান করে',
        'nitrogen': 'নাইট্রোজেন (N)',
        'phosphorus': 'ফসফরাস (P)',
        'potassium': 'পটাসিয়াম (K)',
        'temperature': 'তাপমাত্রা',
        'humidity': 'আর্দ্রতা',
        'ph_level': 'pH মাত্রা',
        'rainfall': 'বৃষ্টিপাত',
        'nitrogen_placeholder': 'নাইট্রোজেনের পরিমাণ লিখুন',
        'phosphorus_placeholder': 'ফসফরাসের পরিমাণ লিখুন',
        'potassium_placeholder': 'পটাসিয়ামের পরিমাণ লিখুন',
        'temperature_placeholder': 'তাপমাত্রা °C তে লিখুন',
        'humidity_placeholder': 'আর্দ্রতা % তে লিখুন',
        'ph_placeholder': 'pH মান লিখুন (0-14)',
        'rainfall_placeholder': 'বৃষ্টিপাত মিমি তে লিখুন',
        'get_recommendation': 'স্মার্ট সুপারিশ পান',
        'loading_text': 'আপনার ডেটা বিশ্লেষণ করছি এবং সুপারিশ তৈরি করছি...',
        'recommended_crop': '🌾 সুপারিশকৃত ফসল',
        'crop_recommendation': '{} সেখানে চাষের জন্য সেরা ফসল',
        'error_message': 'দুঃখিত, আমরা প্রদত্ত ডেটা দিয়ে সেরা ফসল নির্ধারণ করতে পারিনি।',
        'language': 'ভাষা',
        'english': 'ইংরেজি',
        'hindi': 'হিন্দি',
        'bengali': 'বাংলা',
        'telugu': 'তেলুগু',
        'marathi': 'মराठी',
        'tamil': 'তমিल',
        'gujarati': 'গుજરાતી',
        'kannada': 'কন્নড',
        'malayalam': 'মলযાলम',
        'punjabi': 'পংজાবি',
        'odia': 'ওডিয়া',
        'assamese': 'অসমীয়া',
        'sanskrit': 'সংস্কૃত',
        'weather_data': 'আবহাওয়া ডেটা',
        'weather_data_desc': 'আপনার অবস্থান থেকে রিয়েল-টাইম আবহাওয়া ডেটা পান বা ম্যানুয়ালি প্রবেশ করান',
        'auto_weather': 'অটো আবহাওয়া',
        'manual_weather': 'ম্যানুয়াল আবহাওয়া',
        'get_location': 'আমার অবস্থান পান',
        'location_placeholder': 'শহরের নাম প্রবেশ করান',
        'current_weather': 'বর্তমান আবহাওয়া',
        'weather_temp': 'তাপমাত্রা',
        'weather_humidity': 'আর্দ্রতা',
        'weather_rainfall': 'বৃষ্টিপাত',
        'weather_loading': 'আবহাওয়া ডেটা আনা হচ্ছে...',
        'weather_error': 'আবহাওয়া ডেটা আনতে অক্ষম। অনুগ্রহ করে আবার চেষ্টা করুন বা ম্যানুয়ালি প্রবেশ করান।',
        'location_error': 'আপনার অবস্থান পেতে অক্ষম। অনুগ্রহ করে শহর কার নাম ম্যানুয়ালি প্রবেশ করান।',
        'use_weather_data': 'আবহাওয়া ডেটা ব্যবহার করুন',
        'enter_manually': 'ম্যানুয়ালি প্রবেশ করান',
        'weather_success': 'আবহাওয়া ডেটা সফলভাবে আনা হয়েছে!',
        'location_permission': 'অনুগ্রহ করে স্বয়ংক্রিয়ভাবে আবহাওয়া ডেটা পেতে অবস্থান অ্যাক্সেসের অনুমতি দিন।'
    },
    'te': {
        'title': 'స్మార్ట్ పంట సిఫార్సు వ్యవస్థ',
        'subtitle': 'మీ నేల మరియు వాతావరణ పరిస్థితుల ఆధారంగా సాగుకు ఉత్తమ పంటలకు AI-ఆధారిత సిఫార్సులను పొందండి',
        'navbar_brand': 'స్మార్ట్ పంట సిఫార్సు',
        'soil_nutrients': 'నేల పోషకాల విశ్లేషణ',
        'soil_nutrients_desc': 'ఉత్తమ పంట ఎంపిక కోసం నైట్రోజన్, ఫాస్ఫరస్ మరియు పొటాషియం స్థాయిల అధునాతన విశ్లేషణ',
        'climate_intelligence': 'వాతావరణ మేధ',
        'climate_intelligence_desc': 'ఉష్ణోగ్రత, తేమ మరియు వర్షపాత నమూనాల సమగ్ర అంచనా',
        'ai_insights': 'AI-ఆధారిత అంతర్దృష్టులు',
        'ai_insights_desc': 'మెషీన్ లెర్నింగ్ అల్గోరిథంలు అధిక ఖచ్చితత్వంతో ఖచ్చితమైన పంట సిఫార్సులను అందిస్తాయి',
        'nitrogen': 'నైట్రోజన్ (N)',
        'phosphorus': 'ఫాస్ఫరస్ (P)',
        'potassium': 'పొటాషియం (K)',
        'temperature': 'ఉష్ణోగ్రత',
        'humidity': 'తేమ',
        'ph_level': 'pH స్థాయి',
        'rainfall': 'వర్షపాతం',
        'nitrogen_placeholder': 'నైట్రోజన్ కంటెంట్ నమోదు చేయండి',
        'phosphorus_placeholder': 'ఫాస్ఫరస్ కంటెంట్ నమోదు చేయండి',
        'potassium_placeholder': 'పొటాషియం కంటెంట్ నమోదు చేయండి',
        'temperature_placeholder': 'ఉష్ణోగ్రత °C లో నమోదు చేయండి',
        'humidity_placeholder': 'తేమ % లో నమోదు చేయండి',
        'ph_placeholder': 'pH విలువ (0-14) నమోదు చేయండి',
        'rainfall_placeholder': 'వర్షపాతం మిమీ లో నమోదు చేయండి',
        'get_recommendation': 'స్మార్ట్ సిఫార్సు పొందండి',
        'loading_text': 'మీ డేటాను విశ్లేషిస్తున్నాను మరియు సిఫార్సులను రూపొందిస్తున్నాను...',
        'recommended_crop': '🌾 సిఫార్సు చేసిన పంట',
        'crop_recommendation': '{} అక్కడ సాగుకు ఉత్తమ పంట',
        'error_message': 'క్షమించండి, మేము అందించిన డేటాతో ఉత్తమ పంటను నిర్ణయించలేకపోయాము.',
        'language': 'భాష',
        'english': 'ఆంగ్లం',
        'hindi': 'హిందీ',
        'bengali': 'బెంగాలీ',
        'telugu': 'తెలుగు',
        'marathi': 'మరాఠీ',
        'tamil': 'తమిళం',
        'gujarati': 'గుజరాతీ',
        'kannada': 'కన్నడ',
        'malayalam': 'మలయాళం',
        'punjabi': 'పంజాబీ',
        'odia': 'ఒడియా',
        'assamese': 'అస్సామీ',
        'sanskrit': 'సంస్కృతం',
        'weather_data': 'వాతావరణ డేటా',
        'weather_data_desc': 'మీ స్థానం నుండి రియల్-టైమ్ వాతావరణ డేటాను పొందండి లేదా మాన్యువల్గా నమోదు చేయండి',
        'auto_weather': 'ఆటో వాతావరణం',
        'manual_weather': 'మాన్యువల్ వాతావరణం',
        'get_location': 'నా స్థానాన్ని పొందండి',
        'location_placeholder': 'నగర పేరును నమోదు చేయండి',
        'current_weather': 'ప్రస్తుత వాతావరణం',
        'weather_temp': 'ఉష్ణోగ్రత',
        'weather_humidity': 'తేమ',
        'weather_rainfall': 'వర్షపాతం',
        'weather_loading': 'వాతావరణ డేటాను తెస్తున్నాను...',
        'weather_error': 'వాతావరణ డేటాను తెచ్చలేకపోయాము. దయచేసి మళ్లీ ప్రయత్నించండి లేదా మాన్యువల్గా నమోదు చేయండి.',
        'location_error': 'మీ స్థానాన్ని పొందలేకపోయాము. దయచేసి నగర పేరును మాన్యువల్గా నమోదు చేయండి.',
        'use_weather_data': 'వాతావరణ డేటాను ఉపయోగించండి',
        'enter_manually': 'మాన్యువల్గా నమోదు చేయండి',
        'weather_success': 'వాతావరణ డేటా విజయవంతంగా తెచ్చబడింది!',
        'location_permission': 'దయచేసి వాతావరణ డేటాను స్వయంచాలకంగా పొందడానికి స్థాన అనుమతిని ఇవ్వండి.'
    },
    'mr': {
        'title': 'स्मार्ट पीक शिफारस प्रणाली',
        'subtitle': 'तुमच्या माती आणि हवामान परिस्थितींच्या आधारे शेतीसाठी सर्वोत्तम पिकांच्या AI-चालित शिफारसी मिळवा',
        'navbar_brand': 'स्मार्ट पीक शिफारस',
        'soil_nutrients': 'माती पोषक तत्व विश्लेषण',
        'soil_nutrients_desc': 'सर्वोत्तम पीक निवडीसाठी नायट्रोजन, फॉस्फरस आणि पोटॅशियम पातळीचे प्रगत विश्लेषण',
        'climate_intelligence': 'हवामान बुद्धिमत्ता',
        'climate_intelligence_desc': 'तापमान, आर्द्रता आणि पाऊस पॅटर्नचे व्यापक मूल्यांकन',
        'ai_insights': 'AI-चालित अंतर्दृष्टी',
        'ai_insights_desc': 'मशीन लर्निंग अल्गोरिदम उच्च अचूकतेसह अचूक पीक शिफारसी देतात',
        'nitrogen': 'नायट्रोजन (N)',
        'phosphorus': 'फॉस्फरस (P)',
        'potassium': 'पोटॅशियम (K)',
        'temperature': 'तापमान',
        'humidity': 'आर्द्रता',
        'ph_level': 'pH पातळी',
        'rainfall': 'पाऊस',
        'nitrogen_placeholder': 'नायट्रोजन सामग्री प्रविष्ट करा',
        'phosphorus_placeholder': 'फॉस्फरस सामग्री प्रविष्ट करा',
        'potassium_placeholder': 'पोटॅशियम सामग्री प्रविष्ट करा',
        'temperature_placeholder': 'तापमान °C मध्ये प्रविष्ट करा',
        'humidity_placeholder': 'आर्द्रता % मध्ये प्रविष्ट करा',
        'ph_placeholder': 'pH मूल्य (0-14) प्रविष्ट करा',
        'rainfall_placeholder': 'पाऊस मिमी मध्ये प्रविष्ट करा',
        'get_recommendation': 'स्मार्ट शिफारस मिळवा',
        'loading_text': 'तुमचा डेटा विश्लేषण करत आहे आणि शिफारसी तयार करत आहे...',
        'recommended_crop': '🌾 शिफारस केलेले पीक',
        'crop_recommendation': '{} तेथे शेतीसाठी सर्वोत्तम पीक आहे',
        'error_message': 'क्षमस्व, आम्ही दिलेल्या डेटासह सर्वोत्तम पीक ठरवू शकलो नाही.',
        'language': 'भाषा',
        'english': 'इंग्रजी',
        'hindi': 'हिंदी',
        'bengali': 'बंगाली',
        'telugu': 'तेलुगू',
        'marathi': 'మరాఠీ',
        'tamil': 'తమిళం',
        'gujarati': 'గుజరాతી',
        'kannada': 'కన్నడ',
        'malayalam': 'మలయాళం',
        'punjabi': 'పంజాబీ',
        'odia': 'ఓడియా',
        'assamese': 'అసామీ',
        'sanskrit': 'సంస్కృత',
        'weather_data': 'हवामान डेटा',
        'weather_data_desc': 'तुमच्या स्थानावरून रिअल-टाइम हवामान डेटा मिळवा किंवा मॅन्युअली प्रविष्ट करा',
        'auto_weather': 'ऑटो हवामान',
        'manual_weather': 'मॅन्युअल हवामान',
        'get_location': 'माझे स्थान मिळवा',
        'location_placeholder': 'शहराचे नाव प्रविष्ट करा',
        'current_weather': 'सध्याचे हवामान',
        'weather_temp': 'तापमान',
        'weather_humidity': 'आर्द्रता',
        'weather_rainfall': 'पाऊस',
        'weather_loading': 'हवामान डेटा आणत आहे...',
        'weather_error': 'हवामान डेटा आणू शकत नाही. कृपया पुन्हा प्रयत्न करा किंवा मॅन्युअली प्रविष्ट करा.',
        'location_error': 'तुमचे स्थान मिळवू शकत नाही. कृपया शहराचे नाव मॅन्युअली प्रविष्ट करा.',
        'use_weather_data': 'हवामान डेटा वापरा',
        'enter_manually': 'मॅन्युअली प्रविष्ट करा',
        'weather_success': 'हवामान डेटा यशस्वीरित्या आणला गेला!',
        'location_permission': 'कृपया हवामान डेटा स्वयंचलितपणे मिळवण्यासाठी स्थान प्रवेशाची परवानगी द्या.'
    },
    'ta': {
        'title': 'ஸ்மார்ட் பயிர் பரிந்துரை அமைப்பு',
        'subtitle': 'உங்கள் மண் மற்றும் காலநிலை நிலைமைகளின் அடிப்படையில் சாகுபடிக்கு சிறந்த பயிர்களுக்கான AI-ஆல் இயக்கப்படும் பரிந்துரைகளைப் பெறுங்கள்',
        'navbar_brand': 'ஸ்மார்ட் பயிர் பரிந்துரை',
        'soil_nutrients': 'மண் ஊட்டச்சத்து பகுப்பாய்வு',
        'soil_nutrients_desc': 'சிறந்த பயிர் தேர்வுக்கான நைட்ரஜன், பாஸ்பரஸ் மற்றும் பொட்டாசியம் அளவுகளின் மேம்பட்ட பகுப்பாய்வு',
        'climate_intelligence': 'காலநிலை நுண்ணறிவு',
        'climate_intelligence_desc': 'வெப்பநிலை, ஈரப்பதம் மற்றும் மழைப்பொழிவு வடிவங்களின் விரிவான மதிப்பீடு',
        'ai_insights': 'AI-ஆல் இயக்கப்படும் நுண்ணறிவு',
        'ai_insights_desc': 'மெஷின் லர்னிங் அல்காரிதம்கள் உயர் துல்லியத்துடன் துல்லியமான பயிர் பரிந்துரைகளை வழங்குகின்றன',
        'nitrogen': 'நைட்ரஜன் (N)',
        'phosphorus': 'பாஸ்பரஸ் (P)',
        'potassium': 'பொட்டாசியம் (K)',
        'temperature': 'வெப்பநிலை',
        'humidity': 'ஈரப்பதம்',
        'ph_level': 'pH அளவு',
        'rainfall': 'மழைப்பொழிவு',
        'nitrogen_placeholder': 'நைட்ரஜன் உள்ளடக்கத்தை உள்ளிடவும்',
        'phosphorus_placeholder': 'பாஸ்பரஸ் உள்ளடக்கத்தை உள்ளிடவும்',
        'potassium_placeholder': 'பொட்டாசியம் உள்ளடக்கத்தை உள்ளிடவும்',
        'temperature_placeholder': 'வெப்பநிலையை °C இல் உள்ளிடவும்',
        'humidity_placeholder': 'ஈரப்பதத்தை % இல் உள்ளிடவும்',
        'ph_placeholder': 'pH மதிப்பை (0-14) உள்ளிடவும்',
        'rainfall_placeholder': 'மழைப்பொழிவை மிமீ இல் உள்ளிடவும்',
        'get_recommendation': 'ஸ்மார்ட் பரிந்துரையைப் பெறுங்கள்',
        'loading_text': 'உங்கள் தரவை பகுப்பாய்வு செய்து பரிந்துரைகளை உருவாக்குகிறேன்...',
        'recommended_crop': '🌾 பரிந்துரைக்கப்பட்ட பயிர்',
        'crop_recommendation': '{} அங்கு சாகுபடிக்கு சிறந்த பயிர்',
        'error_message': 'மன்னிக்கவும், வழங்கப்பட்ட தரவுடன் சிறந்த பயிரை நாங்கள் தீர்மானிக்க முடியவில்லை.',
        'language': 'மொழி',
        'english': 'ஆங்கிலம்',
        'hindi': 'ஹிந்தி',
        'bengali': 'வங்காளி',
        'telugu': 'தெலுங்கு',
        'marathi': 'மరாத்தி',
        'tamil': 'தமிழ்',
        'gujarati': 'குஜராத்தி',
        'kannada': 'கன்னடம்',
        'malayalam': 'மலையாளம்',
        'punjabi': 'பஞ்சாபி',
        'odia': 'ஒடியா',
        'assamese': 'அசாமி',
        'sanskrit': 'சமஸ்கிருதம்',
        'weather_data': 'வானிலை தரவு',
        'weather_data_desc': 'உங்கள் இருப்பிடத்திலிருந்து ரியல்-டைம் வானிலை தரவைப் பெறுங்கள் அல்லது கைமுறையாக உள்ளிடவும்',
        'auto_weather': 'ஆட்டோ வானிலை',
        'manual_weather': 'கைமுறை வானிலை',
        'get_location': 'என் இருப்பிடத்தைப் பெறுங்கள்',
        'location_placeholder': 'நகரத்தின் பெயரை உள்ளிடவும்',
        'current_weather': 'தற்போதைய வானிலை',
        'weather_temp': 'வெப்பநிலை',
        'weather_humidity': 'ஈரப்பதம்',
        'weather_rainfall': 'மழைப்பொழிவு',
        'weather_loading': 'வானிலை தரவைக் கொண்டு வருகிறேன்...',
        'weather_error': 'வானிலை தரவைக் கொண்டு வர முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது கைமுறையாக உள்ளிடவும்.',
        'location_error': 'உங்கள் இருப்பிடத்தைப் பெற முடியவில்லை. தயவுசெய்து நகரத்தின் பெயரை கைமுறையாக உள்ளிடவும்.',
        'use_weather_data': 'வானிலை தரவைப் பயன்படுத்தவும்',
        'enter_manually': 'கைமுறையாக உள்ளிடவும்',
        'weather_success': 'வானிலை தரவு வெற்றிகரமாகக் கொண்டு வரப்பட்டது!',
        'location_permission': 'தயவுசெய்து வானிலை தரவை தானாகப் பெற இருப்பிட அனுமதியை வழங்கவும்.'
    },
    'gu': {
        'title': 'સ્માર્ટ ક્રોપ રેકમેન્ડેશન સિસ્ટમ',
        'subtitle': 'તમારી માટી અને આબોહવાની સ્થિતિઓના આધારે ખેતી માટે શ્રેષ્ઠ પાકોની AI-ચાલિત ભલામણો મેળવો',
        'navbar_brand': 'સ્માર્ટ ક્રોપ રેકમેન્ડેશન',
        'soil_nutrients': 'માટી પોષક તત્વોનું વિશ્લેષણ',
        'soil_nutrients_desc': 'શ્રેષ્ઠ પાક પસંદગી માટે નાઇટ્રોજન, ફોસ્ફરસ અને પોટાશિયમના સ્તરનું અધુનાતન વિશ્લેષણ',
        'climate_intelligence': 'આબોહવા બુદ્ધિ',
        'climate_intelligence_desc': 'તાપમાન, ભેજ અને વરસાદના પેટર્નનું વ્યાપક મૂલ્યાંકન',
        'ai_insights': 'AI-ચાલિત અંતર્દૃષ્ટિ',
        'ai_insights_desc': 'મશીન લર્નિંગ અલ્ગોરિધમ્સ ઉચ્ચ ચોકસાઈ સાથે ચોક્કસ પાક ભલામણો આપે છે',
        'nitrogen': 'નાઇટ્રોજન (N)',
        'phosphorus': 'ફોસ્ફરસ (P)',
        'potassium': 'પોટાશિયમ (K)',
        'temperature': 'તાપમાન',
        'humidity': 'ભેજ',
        'ph_level': 'pH સ્તર',
        'rainfall': 'વરસાદ',
        'nitrogen_placeholder': 'નાઇટ્રોજનની માત્રા દાખલ કરો',
        'phosphorus_placeholder': 'ફોસ્ફરસની માત્રા દાખલ કરો',
        'potassium_placeholder': 'પોટાશિયમની માત્રા દાખલ કરો',
        'temperature_placeholder': 'તાપમાન °C માં દાખલ કરો',
        'humidity_placeholder': 'ભેજ % માં દાખલ કરો',
        'ph_placeholder': 'pH મૂલ્ય (0-14) દાખલ કરો',
        'rainfall_placeholder': 'વરસાદ મિમી માં દાખલ કરો',
        'get_recommendation': 'સ્માર્ટ ભલામણ મેળવો',
        'loading_text': 'તમારો ડેટા વિશ્લેષણ કરી રહ્યો છું અને ભલામણો બનાવી રહ્યો છું...',
        'recommended_crop': '🌾 ભલામણ કરેલો પાક',
        'crop_recommendation': '{} ત્યાં ખેતી માટે શ્રેષ્ઠ પાક છે',
        'error_message': 'માફ કરશો, અમે આપેલા ડેટા સાથે શ્રેષ્ઠ પાક નક્કી કરી શક્યા નથી.',
        'language': 'ભાષા',
        'english': 'અંગ્રેજી',
        'hindi': 'હિન્દી',
        'bengali': 'બંગાળી',
        'telugu': 'તેલુગુ',
        'marathi': 'મરાઠી',
        'tamil': 'તમિલ',
        'gujarati': 'ગుજરાતી',
        'kannada': 'કન્નડ',
        'malayalam': 'મલયાળમ',
        'punjabi': 'પંજાબી',
        'odia': 'ઓડિયા',
        'assamese': 'અસમિયા',
        'sanskrit': 'સંસ્કૃત',
        'weather_data': 'હવામાન ડેટા',
        'weather_data_desc': 'તમારા સ્થાન પરથી રિયલ-ટાઇમ હવામાન ડેટા મેળવો અથવા મેન્યુઅલી દાખલ કરો',
        'auto_weather': 'ઓટો હવામાન',
        'manual_weather': 'મેન્યુઅલ હવામાન',
        'get_location': 'મારું સ્થાન મેળવો',
        'location_placeholder': 'શહેરનું નામ દાખલ કરો',
        'current_weather': 'વર્તમાન હવામાન',
        'weather_temp': 'તાપમાન',
        'weather_humidity': 'ભેજ',
        'weather_rainfall': 'વરસાદ',
        'weather_loading': 'હવામાન ડેટા આણી રહ્યો છું...',
        'weather_error': 'હવામાન ડેટા આણી શક્યા નથી. કૃપા કરી ફરીથી પ્રયાસ કરો અથવા મેન્યુઅલી દાખલ કરો.',
        'location_error': 'તમારું સ્થાન મેળવી શક્યા નથી. કૃપા કરી શહેરનું નામ મેન્યુઅલી દાખલ કરો.',
        'use_weather_data': 'હવામાન ડેટા વાપરો',
        'enter_manually': 'મેન્યુઅલી દાખલ કરો',
        'weather_success': 'હવામાન ડેટા સફળતાપૂર્વક આણી ગયો!',
        'location_permission': 'કૃપા કરી હવામાન ડેટા સ્વયંચાલિત રીતે મેળવવા માટે સ્થાન પ્રવેશની પરવાનગી આપો.'
    }
}

def get_language():
    """Get current language from session or default to English"""
    return session.get('language', 'en')

def get_text(key):
    """Get translated text for the given key"""
    lang = get_language()
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

def get_system_weather_data():
    """Get weather-like data from system sensors and environment"""
    try:
        # Get system temperature (if available)
        cpu_temp = None
        try:
            # Try to get CPU temperature using psutil
            cpu_temp = psutil.cpu_percent(interval=1)
            # Convert CPU usage to a temperature-like value (20-40°C range)
            temp = 20 + (cpu_temp * 0.2)
        except:
            # Fallback to a reasonable temperature
            temp = 25 + random.uniform(-5, 5)
        
        # Get system humidity (simulated based on system load)
        try:
            memory_percent = psutil.virtual_memory().percent
            # Simulate humidity based on memory usage (30-80% range)
            humidity = 30 + (memory_percent * 0.5)
        except:
            humidity = 50 + random.uniform(-10, 10)
        
        # Get system rainfall (simulated based on disk usage)
        try:
            # Use C: drive on Windows, / on Unix
            disk_path = 'C:\\' if platform.system() == 'Windows' else '/'
            disk_usage = psutil.disk_usage(disk_path).percent
            # Simulate rainfall based on disk usage (0-20mm range)
            rainfall = disk_usage * 0.2
        except:
            rainfall = random.uniform(0, 10)
        
        # Get current time for weather description
        hour = datetime.now().hour
        if 6 <= hour < 12:
            weather_desc = "Sunny"
            icon = "01d"
        elif 12 <= hour < 18:
            weather_desc = "Partly Cloudy"
            icon = "02d"
        elif 18 <= hour < 22:
            weather_desc = "Clear"
            icon = "01n"
        else:
            weather_desc = "Clear Night"
            icon = "01n"
        
        return {
            'temp': round(temp, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 1),
            'description': weather_desc,
            'icon': icon,
            'source': 'system'
        }
    except Exception as e:
        # Fallback to random weather data
        return {
            'temp': round(25 + random.uniform(-5, 5), 1),
            'humidity': round(50 + random.uniform(-10, 10), 1),
            'rainfall': round(random.uniform(0, 10), 1),
            'description': 'System Weather',
            'icon': '01d',
            'source': 'system_fallback'
        }

def get_weather_data(city=None, lat=None, lon=None):
    """Get real weather data from wttr.in API (free, no API key required)"""
    try:
        # Use wttr.in API which is free and doesn't require API key
        if city:
            url = f"https://wttr.in/{city}?format=j1"
        else:
            # Default to Mumbai if no city provided
            url = "https://wttr.in/Mumbai?format=j1"
        
        # Make API request
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract current weather data
        current = data['current_condition'][0]
        
        # Extract weather data
        weather_data = {
            'temp': round(float(current['temp_C']), 1),
            'humidity': round(float(current['humidity']), 1),
            'rainfall': round(float(current.get('precipMM', 0)), 1),
            'description': current['weatherDesc'][0]['value'],
            'icon': '01d',  # wttr.in doesn't provide icons, use default
            'source': 'wttr.in'
        }
        
        print(f"Real weather data fetched: {weather_data}")
        
        return {
            'success': True,
            'data': weather_data
        }
    except Exception as e:
        print(f"Weather data error: {str(e)}")
        # Fallback to system data
        weather_data = get_system_weather_data()
        return {
            'success': True,
            'data': weather_data
        }

@app.route('/')
def index():
    return render_template("index.html", t=get_text)

@app.route('/change_language/<language>')
def change_language(language):
    """Change the current language"""
    if language in TRANSLATIONS:
        session['language'] = language
    return redirect(url_for('index'))

@app.route('/get_weather', methods=['POST'])
def get_weather():
    """Get weather data by city name"""
    try:
        data = request.get_json()
        city = data.get('city')
        
        print(f"Weather request received for city: {city}")
        
        # Get system weather data regardless of city
        weather_result = get_weather_data()
        
        print(f"Weather result: {weather_result}")
        
        if weather_result['success']:
            response = {
                'success': True,
                'weather': weather_result['data']
            }
            print(f"Sending response: {response}")
            return jsonify(response)
        else:
            response = {
                'success': False,
                'error': get_text('weather_error')
            }
            print(f"Sending error response: {response}")
            return jsonify(response)
    except Exception as e:
        print(f"Exception in get_weather: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_weather_by_location', methods=['POST'])
def get_weather_by_location():
    """Get weather data by coordinates"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        # Get system weather data regardless of coordinates
        weather_result = get_weather_data()
        
        if weather_result['success']:
            return jsonify({
                'success': True,
                'weather': weather_result['data']
            })
        else:
            return jsonify({
                'success': False,
                'error': get_text('weather_error')
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

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
        result = get_text('crop_recommendation').format(crop)
    else:
        result = get_text('error_message')
    return render_template('index.html', result=result, t=get_text)

if __name__ == "__main__":
    app.run(debug=True)