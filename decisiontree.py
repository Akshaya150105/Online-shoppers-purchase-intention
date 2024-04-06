import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("online_shoppers_intention.csv")  

#split the dataframe into target variable and affecting features
X = data.drop('Revenue', axis=1)  
y = data['Revenue']  

#categorical variable encoding then train and test the data
X_encoded = pd.get_dummies(X, columns=['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType'])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model :", accuracy)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))
clf.fit(X_train, y_train)



# Make predictions on new data
new_data = pd.DataFrame({
    'Administrative': [3],
    'Administrative_Duration': [87],
    'Informational': [0],
    'Informational_Duration': [0],
    'ProductRelated': [27],
    'ProductRelated_Duration': [798],
    'BounceRates': [0],
    'ExitRates': [0.012644],
    'PageValues': [22.91604],
    'SpecialDay': [0.8],
    'Month': ['Feb'],
    'OperatingSystems': [2],
    'Browser': [2],
    'Region': [3],
    'TrafficType': [1],
    'VisitorType': ['Returning_Visitor'],
    'Weekend': [False]
})
new_data_encoded = pd.get_dummies(new_data, columns=['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType'])

# Ensure that the new dataset has the same columns as the original dataset after encoding
missing_cols = set(X_encoded.columns) - set(new_data_encoded.columns)
for col in missing_cols:
    new_data_encoded[col] = 0  
new_data_encoded = new_data_encoded[X_encoded.columns]

prediction = clf.predict(new_data_encoded)
print("Revenue:", prediction)
