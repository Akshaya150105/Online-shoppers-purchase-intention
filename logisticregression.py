import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("online_shoppers_intention.csv")

#split the dataframe into target variable and affecting features
X = data.drop('Revenue', axis=1)  
y = data['Revenue']  

#categorical variable encoding then train and test the data
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Get feature importance (coefficients)
feature_importance = pd.DataFrame(data=model.coef_[0], index=X_encoded.columns, columns=['Coefficient'])
# Sort feature importance by absolute coefficient values
feature_importance_sorted = feature_importance.reindex(feature_importance.abs().sort_values(by='Coefficient', ascending=False).index)


print("Feature Importance:")
print(feature_importance_sorted)
