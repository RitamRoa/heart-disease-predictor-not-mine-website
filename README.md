# Heart Disease Predictor

This is a simple web application that predicts the likelihood of a person having heart disease based on several health-related factors. The prediction is made using a machine learning model trained on the "Heart 2020" dataset.

## Features

*   Input various health metrics like age, BMI, sex, and lifestyle habits.
*   Get a confidence score (in %) of the likelihood of having heart disease.
*   See the factors that contributed to the prediction.

## Project Structure

```
.
├── app.py                  # Main Flask application
├── model.py                # Script to train the ML model
├── requirements.txt        # Python dependencies
├── heart_2020_cleaned.csv  # Dataset used for training
├── templates/
│   ├── index.html          # Home page with input form
│   └── result.html         # Page to display prediction results
├── static/
│   └── style.css           # CSS for styling
├── .venv/                  # Python virtual environment
└── README.md
```

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/RitamRoa/heart-disease-predictor-not-mine-website.git
cd heart-disease-predictor-not-mine-website
```

### 2. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**On macOS/Linux:**

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

**On Windows:**

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

Once activated, your terminal prompt should be prefixed with `(.venv)`.

### 3. Install Dependencies

Install all the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Train the Model

Before running the application for the first time, you need to train the machine learning model. This will create the `heart_disease_model.pkl` and `model_columns.pkl` files.

```bash
python model.py
```

## Running the Application

Once the setup is complete, you can run the Flask web server:

```bash
flask run
```

Or directly using Python:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`. Open this URL in your web browser.

## Usage

1.  Fill in the form with the required details (Age, Height, Weight, etc.).
2.  Click the "Predict" button.
3.  The next page will show you the predicted confidence score of having heart disease and the reasons behind the score.
