# Enhanced Loan Eligibility Prediction System

A modern, production-ready Flask web application for predicting loan eligibility using machine learning. This system features a clean architecture, modern UI, secure authentication, and is optimized for both local development and cloud deployment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## Features

### Core Functionality
- **Advanced ML Prediction**: RandomForest model with hyperparameter optimization
- **Feature Engineering**: Advanced feature engineering including CIBIL scores, loan-to-income ratios
- **Real-time Predictions**: Instant loan eligibility predictions with confidence scores

### User Experience
- **Modern UI**: Responsive design with Tailwind CSS
- **Secure Authentication**: User registration and login with password hashing
- **Admin Dashboard**: Administrative panel for viewing all predictions
- **Mobile Responsive**: Works seamlessly on all devices

### Technical Excellence
- **SQLite Database**: Lightweight, deployment-friendly database
- **RESTful API**: Clean API endpoints for frontend integration
- **Error Handling**: Comprehensive error handling and logging
- **Health Monitoring**: Built-in health check endpoints
- **Security**: Secure session management and password hashing

### Deployment Ready
- **Vercel Compatible**: Optimized for serverless deployment
- **Environment Configuration**: Flexible environment-based configuration
- **Production Optimized**: Ready for production deployment

## Prerequisites

Before running this application, ensure you have:

- **Python 3.8+** installed on your system
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Node.js** (optional, for Vercel CLI deployment)

## Local Development Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/aaronseq12/Loan-eligibility-calculator.git
cd Loan-eligibility-calculator
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv loan_env

# Activate virtual environment
# On Windows:
loan_env\Scripts\activate
# On macOS/Linux:
source loan_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 4: Environment Configuration

```bash
# Create environment file from template
cp .env.example .env

# Edit the .env file with your preferred settings
# The default values should work for local development
```

### Step 5: Initialize the Database

The application will automatically create the SQLite database and tables on first run. No manual database setup required!

### Step 6: Train the ML Model (Optional)

```bash
# Navigate to models directory
cd models

# Run the model trainer (this will take a few minutes)
python model_trainer.py

# Return to main directory
cd ..
```

**Note**: Pre-trained models are included in the repository, so this step is optional for testing.

### Step 7: Run the Application

```bash
# Start the Flask application
python app.py
```

The application will be available at: **http://localhost:5000**

## Using the Application

### For Users

1. **Registration**: Navigate to `/signup` to create a new account
2. **Login**: Use your credentials to log in at the home page
3. **Prediction**: Fill out the loan application form in the dashboard
4. **Results**: View your loan eligibility prediction with confidence score

### For Administrators

1. **Admin Login**: Navigate to `/admin-login`
2. **Default Credentials**: 
   - Username: `admin`
   - Password: `admin123`
3. **Dashboard**: View all user predictions and statistics

### API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/login` - User authentication
- `POST /api/register` - User registration
- `POST /api/predict` - Loan prediction
- `POST /api/admin/login` - Admin authentication

## Model Information

### Algorithm
- **Primary Model**: Random Forest Classifier
- **Hyperparameter Optimization**: Grid Search with Cross-Validation
- **Feature Engineering**: Advanced feature engineering with 18+ features

### Features Used
- Applicant Income (log-transformed)
- Co-applicant Income (log-transformed)
- Loan Amount (log-transformed)
- Loan Term
- Credit History
- CIBIL Score
- Loan-to-Income Ratio
- EMI Feature
- Demographic factors (Gender, Marital Status, Dependents, Education)
- Employment Status
- Property Area

### Performance Metrics
- **Accuracy**: ~85-90% on test data
- **Cross-Validation Score**: Optimized through grid search
- **Confidence Scoring**: Probability-based confidence levels

## Deployment

### Vercel Deployment (Recommended)

1. **Install Vercel CLI**
```bash
npm install -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Deploy**
```bash
# Deploy to production
vercel --prod
```

4. **Set Environment Variables**
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add `SECRET_KEY` with a secure random string
   - Add `FLASK_ENV=production`

### Other Deployment Options

The application is compatible with:
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **Railway**: Direct deployment from GitHub
- **DigitalOcean App Platform**: Compatible with Python buildpack
- **AWS Lambda**: Can be adapted for serverless deployment

## Project Structure

```
Loan-eligibility-calculator/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── vercel.json                    # Vercel deployment config
├── .env.example                   # Environment variables template
├── README.md                      # This file
│
├── models/                        # ML models and training
│   ├── model_trainer.py          # Enhanced model training script
│   ├── loan_eligibility_model.pkl # Trained model (auto-generated)
│   ├── feature_scaler.pkl        # Feature scaler (auto-generated)
│   └── feature_columns.pkl       # Feature columns (auto-generated)
│
├── templates/                     # HTML templates
│   ├── base.html                 # Base template with modern design
│   ├── auth/
│   │   ├── login.html           # User login page
│   │   └── signup.html          # User registration page
│   ├── dashboard/
│   │   └── home.html            # User dashboard
│   └── admin/
│       ├── login.html           # Admin login page
│       └── dashboard.html       # Admin dashboard
│
├── static/                        # Static files
│   ├── css/                      # Custom stylesheets
│   ├── js/
│   │   └── common.js            # Common JavaScript utilities
│   └── images/                   # Application images
│
└── data/                         # Training data and notebooks
    ├── train.csv                 # Training dataset
    ├── test.csv                  # Test dataset
    └── *.ipynb                   # Jupyter notebooks
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Application Configuration
FLASK_ENV=development
SECRET_KEY=your-super-secret-key-here
PORT=5000

# Database Configuration
DATABASE_PATH=loan_prediction.db

# Model Configuration
MODEL_PATH=models
```

### Security Configuration

- **Secret Key**: Generate a secure secret key for session management
- **Password Hashing**: Uses Werkzeug's secure password hashing
- **Session Security**: Secure session configuration for production

## Testing

### Manual Testing

1. **User Registration and Login**
   - Test user registration with various inputs
   - Verify login functionality
   - Test password validation

2. **Loan Prediction**
   - Test with different loan scenarios
   - Verify prediction accuracy
   - Test edge cases and invalid inputs

3. **Admin Functionality**
   - Test admin login
   - Verify prediction viewing
   - Test admin dashboard functionality

### API Testing

Use tools like Postman or curl to test API endpoints:

```bash
# Health check
curl http://localhost:5000/api/health

# User registration
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"user_id":"testuser","email":"test@example.com","full_name":"Test User","password":"password123","confirm_password":"password123"}'
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 Python style guide
- Add proper error handling and logging
- Write comprehensive docstrings
- Test your changes thoroughly
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source loan_env/bin/activate  # macOS/Linux
   loan_env\Scripts\activate    # Windows

   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **Database Issues**
   ```bash
   # Delete existing database and restart
   rm loan_prediction.db
   python app.py
   ```

3. **Model Loading Errors**
   ```bash
   # Retrain the model
   cd models
   python model_trainer.py
   cd ..
   ```

4. **Port Already in Use**
   ```bash
   # Use different port
   export PORT=8000
   python app.py
   ```

### Getting Help

- **Issues**: Create an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact the maintainer for urgent issues

## Performance Monitoring

### Built-in Monitoring

- **Health Check Endpoint**: `/api/health`
- **Application Logging**: Comprehensive logging throughout the application
- **Error Tracking**: Detailed error logging and user feedback

### Production Monitoring

For production deployments, consider integrating:
- **Application Performance Monitoring** (APM) tools
- **Error Tracking** services like Sentry
- **Log Aggregation** services
- **Uptime Monitoring** tools

## Future Enhancements

### Planned Features

- **Enhanced ML Models**: XGBoost, Neural Networks
- **Real-time CIBIL Integration**: Live CIBIL score fetching
- **Document Upload**: Support for income and identity documents
- **Email Notifications**: Automated email updates
- **Advanced Analytics**: Detailed prediction analytics
- **Mobile App**: React Native mobile application
- **API Documentation**: Swagger/OpenAPI documentation

### Technical Improvements

- **Caching**: Redis caching for improved performance
- **Microservices**: Break into microservices architecture
- **Container Support**: Docker containerization
- **CI/CD Pipeline**: Automated testing and deployment
- **Load Testing**: Performance optimization

## 📞 Support

For support and questions:

- **GitHub Issues**: [Create an issue](https://github.com/aaronseq12/Loan-eligibility-calculator/issues)
- **Email**: aaronsequeira12@gmail.com
- **LinkedIn**: Connect for professional inquiries

## 🙏 Acknowledgments

- **Scikit-learn**: For excellent machine learning tools
- **Flask**: For the robust web framework
- **Tailwind CSS**: For beautiful, responsive styling
- **Vercel**: For seamless deployment platform
- **Contributors**: Thanks to all contributors and testers

---

**Built with ❤️ by Aaron Sequeira**

*Happy Coding! 🚀*
