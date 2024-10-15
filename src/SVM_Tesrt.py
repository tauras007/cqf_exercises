import shap
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf

# Sample data loading (e.g., stock data with trend classification)
data = data = yf.Ticker('BTC-USD').history(period="max", interval="1h")

# Feature Engineering: Example technical indicators (moving average)
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Create target variable: 1 for upward trend, 0 for downward trend
data['Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop NaN values that result from technical indicator calculations
data = data.dropna()

# Define features and target
X = data[['SMA_10', 'SMA_50']]  # Add more features if available
y = data['Trend']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (SVM benefits from normalized data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, probability=True)  # Enable probability estimates
svm_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = svm_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Use only a subset of the test set for SHAP
sample_size = 100  # You can adjust this number
X_test_sample = X_test[:sample_size]

# SHAP Kernel Explainer for SVM
# SHAP Kernel Explainer with parallel processing
explainer = shap.KernelExplainer(svm_model.predict_proba, X_train, n_jobs=-1)  # Use all available cores
shap_values = explainer.shap_values(X_test_sample)

# Visualize SHAP values
shap.summary_plot(shap_values, X_test_sample, feature_names=['SMA_10', 'SMA_50'])
