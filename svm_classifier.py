import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE  # For handling imbalanced data

# Load the CSV file
data = pd.read_csv("/Users/shahidibrahim/Desktop/Project-I---Image-Processing/facial_features.csv")  # Adjust path as needed


# Separate features and labels (assuming 'label' is the name of your label column)
if 'label' in data.columns:
    X = data.drop('label', axis=1)  # Feature columns
    y = data['label']  # Labels
else:
    print("The 'label' column is not found in the CSV file.")

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handle imbalanced classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for SVM using GridSearchCV
param_grid = {
    'C': [1, 10],
    'gamma': [0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced', random_state=42), param_grid, refit=True, verbose=2, cv=2)
grid.fit(X_train_scaled, y_train)

# Best parameters from the grid search
print(f"Best Parameters: {grid.best_params_}")

# Use the best model from the grid search for predictions
y_pred = grid.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
