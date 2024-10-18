# Import necessary libraries
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Load the saved models and dataset
rf_model = joblib.load('rf_model.pkl')
nn_model = load_model('nn_model.h5')
scaler = joblib.load('scaler.pkl')
diabetes = pd.read_csv('diabetes_dataset.csv')

# Splitting dataset into features (X) and labels (y)
X = diabetes.drop('Outcome', axis=1).values  # Features
y = diabetes['Outcome'].values  # Labels (Outcome column)

# Scale features
X_scaled = scaler.transform(X)

# Function to visualize the knowledge graph
def visualize_graph():
    G = nx.Graph()
    # Example knowledge graph for diabetes management recommendations
    G.add_edge('Diabetic', 'Regular Exercise', recommendation='30 mins daily')
    G.add_edge('Diabetic', 'Diet Control', recommendation='Low sugar, balanced diet')
    G.add_edge('Not Diabetic', 'Maintain Weight', recommendation='Regular weight checks')

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'recommendation')
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Knowledge Graph: Diabetes Management Recommendations")
    plt.show()

# Visualize the knowledge graph
visualize_graph()

# Predictions for visualization
rf_preds = rf_model.predict(X_scaled)
nn_preds = (nn_model.predict(X_scaled) > 0.5).astype(int).flatten()

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y, rf_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion matrix for Neural Network
cm_nn = confusion_matrix(y, nn_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix (Neural Network)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
