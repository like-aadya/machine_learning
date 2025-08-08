import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Data inside the script (same as fruits.csv content)
data = {
    'Weight': [150, 170, 140, 130, 160, 120],
    'ColorCode': [1, 1, 2, 2, 1, 2],
    'Texture': [0, 0, 1, 1, 0, 1],
    'Label': ['Apple', 'Apple', 'Orange', 'Orange', 'Apple', 'Orange']
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Weight', 'ColorCode', 'Texture']]
y = df['Label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict example
sample = [[155, 1, 0]]  # Weight=155, ColorCode=1, Texture=0
predicted = le.inverse_transform(model.predict(sample))
print(f"Predicted fruit: {predicted[0]}")