# Import necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
diabetes = pd.read_csv(r'C:\Users\saiki\Hands_on\diabetes_dataset.csv')

# Display dataset overview
print(diabetes.head())
print(diabetes.info())

# Visualizing the class distribution (Outcome)
sns.countplot(x='Outcome', data=diabetes)
plt.show()

# Correlation heatmap
sns.heatmap(diabetes.corr(), annot=True)
plt.show()

# Splitting dataset into features (X) and labels (y)
X = diabetes.drop('Outcome', axis=1).values  # Features
y = diabetes['Outcome'].values  # Labels (Outcome column)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Building and training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the Random Forest model
joblib.dump(rf_model, 'rf_model.pkl')

# Building the Sequential Neural Network
nn_model = tf.keras.models.Sequential()

# Input layer and first hidden layer with dropout
nn_model.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(tf.keras.layers.Dropout(0.2))

# Second hidden layer with dropout
nn_model.add(tf.keras.layers.Dense(units=400, activation='relu'))
nn_model.add(tf.keras.layers.Dropout(0.2))

# Output layer (binary classification)
nn_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
nn_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save the Neural Network model
nn_model.save('nn_model.h5')

# Evaluate Random Forest model
rf_preds = rf_model.predict(X_test)
print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report (Random Forest):\n", classification_report(y_test, rf_preds))

# Evaluate Neural Network model
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int).flatten()
print("Neural Network Model Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Classification Report (Neural Network):\n", classification_report(y_test, y_pred_nn))
