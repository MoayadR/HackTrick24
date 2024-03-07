from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

data = pd.read_csv("MlMediumTrainingData.csv")
X_train_x = data['x_']
Y_train_y = data['y_']

X_train = []
for i in range(len(X_train_x)):
    X_train.append([X_train_x[i] , Y_train_y[i]])

y_train = np.array(data['class'])

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test the model with an example point

predicted_label = rf_model.predict(X_train)


# Calculate accuracy
accuracy = accuracy_score(y_train, predicted_label)
print("Accuracy:", accuracy)


