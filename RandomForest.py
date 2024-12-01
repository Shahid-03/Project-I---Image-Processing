import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the CSV file
csv_file_path = '/Users/shahidibrahim/Desktop/Project-I---Image-Processing/facial_features.csv'  # Adjust path as needed
try:
    data = pd.read_csv(csv_file_path)
    print(data.head())  # Print first few rows to confirm data load
except FileNotFoundError:
    print(f"File not found. Please check the path: {csv_file_path}")

# Separate features and labels (assuming 'label' is the name of your label column)
if 'label' in data.columns:
    X = data.drop('label', axis=1)  # Feature columns
    y = data['label']  # Labels
else:
    print("The 'label' column is not found in the CSV file.")

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                    param_distributions=param_dist,
                                    n_iter=10,  # Number of parameter settings to try
                                    cv=5,
                                    n_jobs=-1,
                                    scoring='accuracy',
                                    random_state=42)

# Fit the model
random_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
print("Best parameters found: ", random_search.best_params_)
best_rf_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Improved Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
