# CryptoInsight Pro - Bitcoin Price Prediction Platform

A modern web application that provides advanced Bitcoin price predictions using machine learning algorithms and real-time market analysis.

## Features

- **Advanced Price Predictions**: Utilizes ensemble machine learning models to provide accurate Bitcoin price predictions
- **Technical Analysis**: Comprehensive technical indicators and chart analysis
- **Market Analysis**: Detailed market metrics, correlations, and sentiment analysis
- **Real-time Updates**: Live price updates and market data
- **Modern UI/UX**: Responsive design with interactive charts and animations
- **Multi-page Layout**: Organized sections for predictions, analysis, news, and more

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python, Flask
- **Data Analysis**: pandas, numpy, scikit-learn
- **Data Source**: Yahoo Finance API
- **Visualization**: Chart.js
- **Styling**: Bootstrap 5, Custom CSS

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cryptoinsight-pro.git
   cd cryptoinsight-pro
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## Project Structure

```
cryptoinsight-pro/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
│   ├── base.html     # Base template
│   ├── index.html    # Home page
│   ├── predictions.html  # Predictions page
│   └── analysis.html    # Analysis page
└── README.md         # Project documentation
```

## Features in Detail

### Price Prediction
- Ensemble of machine learning models:
  - Random Forest
  - Gradient Boosting
  - Support Vector Regression
  - Lasso Regression
- Real-time price updates
- Confidence scores for predictions
- Multiple timeframe predictions

### Technical Analysis
- Multiple technical indicators
- Interactive charts
- Support and resistance levels
- Volume analysis
- Moving averages

### Market Analysis
- Market sentiment analysis
- Asset correlations
- News impact analysis
- On-chain metrics

## API Endpoints

- `/api/current_price`: Get current Bitcoin price and metrics
- `/api/historical_data`: Get historical price data
- `/api/predict`: Get price predictions
- `/api/performance_metrics`: Get performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Flask and Bootstrap
- Charts powered by Chart.js 
