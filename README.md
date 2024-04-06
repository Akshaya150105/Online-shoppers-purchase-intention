REVENUE PREDICTION USING DECISION TREE:
                       Predicting whether user session will result in revenue (True or False) based on the various features such as number of pages,duration,bounce rate,exit rate,special days,operating systems etc.
So we used decision tree classsifier to predict the revenue based on the features by using the scikit-learn library
 This code loads the dataset, separates the features and target variable, splits the data into training and testing sets, initializes a decision tree classifier, trains the model on the training data, makes predictions on the test data, and evaluates the model's performance using accuracy and a classification report.

FEATURE IMPORTANCE USING LOGISTIC REGRESSION 
                 Using logistic regression,wwe got the coefficients of each features representing its influence on the probability of the target variable being true or false
LIBRARY USED: scikit-learn library
Positive coefficients indicate that an increase in the feature value leads to an increase in the probability of revenue, while negative coefficients indicate the opposite.
From the output we observe that
PageValues has the highest positive influence on revenue.
Months like November, February, and May positively affect revenue, while others like December and March have negative impacts.
ProductRelated_Duration positively influences revenue, while BounceRates negatively impact it.
