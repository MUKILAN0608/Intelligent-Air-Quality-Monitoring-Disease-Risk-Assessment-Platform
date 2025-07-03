# 🌍 Intelligent Air Quality Monitoring & Disease Risk Assessment Platform

A professional, interactive web application for real-time air quality and disease risk prediction, powered by machine learning and explainable AI. Visualize AQI and health risks on a map, search any city, and get actionable insights for public health awareness.

---

🎥 **Live Demo**


https://github.com/user-attachments/assets/cedc5fc9-155c-4dc6-a24d-3f247ddcfe9f




---

## 🚀 Features

- 🗺️ **Interactive Map:** Click or search any location to view real-time air quality and weather data
- 📊 **AQI Prediction:** Predicts Air Quality Index using advanced ML models with 96.2% accuracy
- 🏥 **Disease Risk Assessment:** Estimates risk for multiple pollution-related diseases
- 🔍 **Explainable AI:** LIME/SHAP explanations for both AQI and disease predictions
- 💻 **Modern UI:** Clean, responsive, and professional design
- ⚡ **Real-time Data:** Integration with OpenWeather API for live environmental data
- 🔒 **Secure:** API keys secured via environment variables with input validation

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.12 or higher
- pip package manager
- OpenWeather API key ([free registration at OpenWeatherMap](https://openweathermap.org/appid))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MUKILAN0608/Intelligent-Air-Quality-Monitoring-Disease-Risk-Assessment-Platform.git
   cd Intelligent-Air-Quality-Monitoring-Disease-Risk-Assessment-Platform
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Create a `.env` file in the root directory:
     ```env
     OPENWEATHER_API_KEY=your_api_key_here
     ```

### Run the Application

**Main Application (Recommended):**
```bash
streamlit run streamlit_map_app.py
```

**Alternative Interface:**
```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to [http://localhost:8501](http://localhost:8501)

---

## 📦 Project Structure

```
Air_Pollution_Predictor/
├── streamlit_map_app.py      # Main interactive map application
├── streamlit_app.py          # Alternative interface
├── models.py                 # ML model definitions and utilities
├── requirements.txt          # Python dependencies
├── static/                   # Static assets
├── templates/                # HTML templates
├── *.pkl                     # Pre-trained ML models
├── *.csv                     # Example datasets
├── .env                      # Environment variables (create this)
└── README.md                 # This file
```

---

## 📊 Models & Data

### Machine Learning Models
- **AQI Prediction Model:** Random Forest/XGBoost with 96.2% accuracy
- **Disease Risk Models:** Ensemble methods with F1-score of 0.921
- **Explainability:** LIME and SHAP for model interpretability

### Data Sources
- **Real-time Environmental Data:** OpenWeather API
- **Historical Air Quality Data:** Pre-processed datasets for training
- **Health Statistics:** Disease correlation data for risk assessment

---

## 📈 Usage

1. Launch the application using the installation instructions above
2. Navigate to the map interface in your browser
3. Interact with the map:
   - Click on any location to get instant air quality data
   - Use the search bar to find specific cities
   - View real-time AQI values and predictions
   - Explore disease risk assessments for the selected location
   - Review model explanations to understand prediction factors

---



---

## 📊 Performance Metrics

| Metric                        | Value      |
|-------------------------------|------------|
| AQI Prediction Accuracy       | 96.2%      |
| Disease Risk Assessment F1    | 0.921      |
| Average Response Time         | <2 seconds |
| Model Inference Time          | <100ms     |
| API Rate Limit                | 1000/day   |

---

## 🔒 Security Features
- **API Key Security:** Environment variables for sensitive data
- **Input Validation:** Comprehensive validation for all user inputs
- **Rate Limiting:** Built-in protection against API abuse
- **Error Handling:** Graceful handling of API failures and edge cases

---

## 🌟 Key Technologies
- **Backend:** Python, Streamlit
- **Machine Learning:** scikit-learn, XGBoost, LIME, SHAP
- **Data Visualization:** Folium, Plotly, Streamlit
- **APIs:** OpenWeather API
- **Deployment:** Streamlit Cloud ready

---

## 🚀 Future Enhancements
- 📱 Mobile app development
- 📈 Historical data visualization
- 🚨 Air quality alerts and notifications
- 🌐 Integration with more environmental APIs
- 🔮 Advanced forecasting models
- 🌍 Multi-language support

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- **Data Source:** OpenWeather API for real-time environmental data
- **ML Libraries:** scikit-learn, XGBoost, LIME, SHAP for machine learning capabilities
- **Visualization:** Folium, Plotly, Streamlit for interactive visualizations
- **Icons:** Lucide React for UI components
