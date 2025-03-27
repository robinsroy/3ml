# Student Performance Prediction System

A Flask-based web application that uses machine learning to predict student performance and provide personalized recommendations. The system integrates three different machine learning models:

1. **Logistic Regression**: Predicts whether a student will pass or fail based on academic factors
2. **K-Nearest Neighbors (KNN)**: Recommends study strategies based on similar students' performance
3. **Polynomial Regression**: Predicts final exam scores using non-linear relationships

## Features

- Interactive web interface with three dedicated pages for each model
- Real-time predictions and recommendations
- Performance metrics display (accuracy and R² scores)
- Responsive design that works on both desktop and mobile devices
- Data preprocessing and normalization
- Model evaluation using train-test splitting

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd student-performance-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Choose a prediction model from the navigation menu:
   - Logistic Regression for pass/fail prediction
   - KNN for study strategy recommendations
   - Polynomial Regression for final exam score prediction

4. Enter the required information in the form and click the predict button to get results.

## Project Structure

```
student-performance-prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/
│   └── css/
│       └── style.css     # Custom CSS styles
└── templates/
    ├── base.html         # Base template
    ├── index.html        # Home page
    ├── logistic.html     # Logistic Regression interface
    ├── knn.html          # KNN interface
    └── polynomial.html   # Polynomial Regression interface
```

## Model Details

### Logistic Regression
- Predicts pass/fail status
- Features: study hours, attendance, previous scores, parental education, mid-term scores
- Output: Binary classification (0: Fail, 1: Pass)

### KNN
- Recommends study strategies
- Features: study hours, attendance, sleep hours, extracurricular activities, mid-term scores
- Output: Performance level (low, medium, high)

### Polynomial Regression
- Predicts final exam scores
- Features: study hours, previous scores, mid-term scores, attendance
- Output: Predicted score (0-100)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 