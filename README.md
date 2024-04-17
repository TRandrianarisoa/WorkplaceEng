# WorkplaceEng


* Environment Setup:
  
Run with Python 3.8 and use Numpy 1.24.3, Pandas 2.0.1, matplotlib 3.7.1, Seaborn 0.13.2, Scipy 1.10.1 and Scikit-Learn 1.3.0
* Data Preparation:

After a first look of the different datasets provided, which can all be linked thanks to the ID ‘EmployeeID’, there are only a handful of missing values in the variables ‘EnvironmentSatisfaction’, ‘JobSatisfaction’, ‘WorkLifeBalance’, ‘NumCompaniesWorked’ and ‘TotalWorkingYears’. For the numerical variables, we impute these values with the observed mean in the corresponding column and, for the categorical variables, we replace with the most frequent observed value.

Three variables are constant, ‘EmployeeCount’, ‘StandardHours’ and ‘Over18’, so we discard them as they bring no information. Finally, we extract from the in\_time and out\_time datasets two new features: the average number of hours an employee worked per day during the year (‘AverageHoursPerday’) and the the number of off days an employee had during the year (‘DaysNotWorked’).

In the hand, we end up with 14 categorical variables (which we encode with OrdinalEncoding in the final model) and 12 numerical variables we standardized. Out of the 4410 samples, we observe 711 positive values for the target ’Attrition’. The problem we have to tackle here is a binary classification with an imbalanced dataset.

* Algorithm Implementation:

The final classification model is a calibrated Histogram  Gradient Boosting Classifier with hyperparameters tuned with a random grid search. 

* Model Evaluation:
  
To measure the quality of the prediction made by our models, we use the F1 score as a metric to focus on the discovery of the positive samples and take into account the fact that the dataset is imbalanced. It is an average of precision and recall.
