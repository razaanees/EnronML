# EnronML
This project trains several machine learning algorithms to identify persons-of-interest in the Enron scandal using the Enron email and financial datasets.

# Details
_poi_id.py_ uses the Naive Bayes algorithm and 5 selected features to identify persons-of-interest (people involved in the corporate fraud in Enron). The features were selected using the "SelectKBest" algorithm from the sklearn package to find the 5 best indicators of involvement in the fraud. According to the dataset, the most indicative features of fraud involvement are:
1. Salary
2. Bonus
3. Exercised stock option
4. Total stock value
5. Deferred income
