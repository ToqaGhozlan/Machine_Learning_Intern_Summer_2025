# Taxi Fare Prediction using Django

A modern web application that predicts taxi fares using machine learning models deployed with Django. The application features a beautiful, responsive UI and uses Ridge Regression and Decision Tree algorithms to provide accurate fare estimates.

## Features

- **Multiple ML Models**: Ridge Regression and Decision Tree for fare prediction
- **Modern UI**: Beautiful, responsive interface with animations and gradients
- **Real-time Predictions**: Instant fare calculations based on trip parameters
- **Mobile Responsive**: Works seamlessly on all devices
- **Interactive Design**: Smooth animations and user-friendly experience

## Tech Stack

- **Backend**: Django 4.x, Python 3.13
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: scikit-learn
- **Database**: SQLite (default)
- **Styling**: Custom CSS with modern animations and gradients

## Prerequisites

Before running this project, make sure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/malakasaber/Model-Deployment-using-Django.git
   cd Model-Deployment-using-Django
   ```

2. **Create and activate virtual environment**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install django scikit-learn pandas numpy
   ```

4. **Run database migrations**

   ```bash
   python manage.py migrate
   ```

5. **Start the development server**

   ```bash
   python manage.py runserver
   ```

6. **Open your browser**
   Navigate to `http://127.0.0.1:8000/` to see the application.

## Usage

1. **Fill in the form** with the following trip details:

   - Pickup and dropoff coordinates (longitude/latitude)
   - Passenger count
   - Distances to key locations (JFK, EWR, LGA airports, Statue of Liberty, NYC)
   - Trip distance and bearing

2. **Click "Predict Fare"** to get instant predictions

3. **View results** in the beautiful modal popup showing predictions from both models

## Project Structure

```
DjangoTask/
├── deploymentproject/          # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── myapp/                      # Main Django app
│   ├── models/                 # Pre-trained ML models
│   │   ├── transformer.pkl
│   │   ├── transformer2.pkl
│   │   ├── XGB.pkl
│   │   ├── Linear.pkl
│   │   ├── KNN.pkl
│   │   ├── ridgeregression.pkl
│   │   └── decisiontree.pkl
│   ├── templates/myapp/        # HTML templates
│   │   └── index.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── ml_pipeline.py          # ML model loading and prediction logic
│   ├── models.py
│   ├── model2.py
│   ├── urls.py
│   ├── views.py
│   └── tests.py
├── db.sqlite3                  # SQLite database
├── manage.py                   # Django management script
└── README.md
```

## Machine Learning Models

### XGBoost

- **Purpose:** Gradient boosting framework optimized for speed and performance
- **Strengths:** High accuracy, handles missing data, effective on structured data
- **Use Case:** Predicts fares with fine-grained performance tuning and scalability

### Linear Regression

- **Purpose:** Models linear relationship between input variables and output
- **Strengths:** Fast, interpretable, works well with linearly correlated data
- **Use Case:** Predicts fares based on distance when the relationship is linear

### K-Nearest Neighbors (KNN)

- **Purpose:** Instance-based learning model for classification and regression
- **Strengths:** Simple, non-parametric, adapts well to local data structure
- **Use Case:** Estimates fares by comparing with similar past trips

### Ridge Regression

- **Purpose**: Linear regression with L2 regularization
- **Strengths**: Good for handling multicollinearity, stable predictions
- **Use Case**: Baseline model for fare prediction

### Decision Tree

- **Purpose**: Non-linear tree-based regression model
- **Strengths**: Captures complex patterns, easy to interpret
- **Use Case**: Handles non-linear relationships in fare calculation

## 📊 Input Features

The model uses the following 20 features for prediction:

| Feature             | Description                           |
| ------------------- | ------------------------------------- |
| `pickup_longitude`  | Pickup location longitude             |
| `pickup_latitude`   | Pickup location latitude              |
| `dropoff_longitude` | Dropoff location longitude            |
| `dropoff_latitude`  | Dropoff location latitude             |
| `passenger_count`   | Number of passengers                  |
| `jfk_dist`          | Distance to JFK Airport               |
| `ewr_dist`          | Distance to Newark Airport            |
| `lga_dist`          | Distance to LaGuardia Airport         |
| `sol_dist`          | Distance to Statue of Liberty         |
| `nyc_dist`          | Distance to NYC center                |
| `distance`          | Trip distance                         |
| `bearing`           | Direction of travel                   |

## UI Features

- **Modern Design**: Gradient backgrounds and glassmorphism effects
- **Responsive Layout**: Grid-based form layout that adapts to screen size
- **Interactive Elements**: Hover effects, smooth transitions, and animations
- **Visual Feedback**: Loading spinners, input validation, and success modals
- **Accessibility**: Proper contrast ratios and semantic HTML

## Development

### Adding New Models

1. Train your model and save it as a `.pkl` file
2. Place it in the `myapp/models/` directory
3. Update `ml_pipeline.py` to load and use your model
4. Modify the view and template to display the new predictions

### Customizing the UI

- Edit `myapp/templates/myapp/index.html` for structure and styling
- Modify CSS classes for different visual themes
- Add new animations or interactive elements as needed

## API Endpoints

- `GET /` - Main page with prediction form
- `POST /` - Submit prediction request and get results

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

**Malak Ahmed Saber**

- GitHub: [@malakasaber](https://github.com/malakasaber)

**Marawan Abbas**

- Github: [@MarwanAbbas205](https://github.com/MarwanAbbas205)

## Acknowledgments

- Built during Cellula Technologies Internship
- Inspired by real-world taxi fare prediction systems
- UI design influenced by modern web design trends

## Support

If you have any questions or run into issues, please:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Contact the maintainer

---

