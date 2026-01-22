import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset
# We use 'data.csv' because you are running this from INSIDE the model folder
df = pd.read_csv('data.csv')

# 2. Feature Selection (Selecting 5 features from your list)
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
X = df[features]
y = df['diagnosis']  # Usually 'M' for Malignant and 'B' for Benign

# 3. Preprocessing
# a. Handling missing values
X = X.fillna(X.mean())

# b. Encoding the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# c. Feature Scaling (MANDATORY for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 5. Implement SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, predictions))

# 7. Save the trained model and preprocessing tools to disk
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nSuccess! Model, Scaler, and Encoder saved inside the model folder.")