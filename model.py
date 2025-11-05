import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('heart_2020_cleaned.csv')

# Preprocessing
# Convert 'Yes'/'No' to 1/0 for the target variable
df['HeartDisease'] = df['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
gb_classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = gb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save the trained model and the columns
joblib.dump(gb_classifier, 'heart_disease_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

print("Model and columns saved.")
