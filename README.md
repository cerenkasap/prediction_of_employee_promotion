## Prediction of Employee Promotion - Binary Classification 🏆: Project Overview

Created a model that can classify an employee promotion with **(89.49% Accuracy)**.

Used two datasets: PromosSample.csv as historical data with **54808 examples** and test.csv as current data **23490 examples** using pandas library in python.

Applied **Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Naive Bayes**, and **KNeighborsClassifier** and optimized using **GridSearchCV** to find the best model.

### Code Used

Python version: *Python 3.7.11* 

Packages: *pandas, o seaborn, matplotlib, numpy, scikit-learn, and SMOTE*

### Resources Used

[Machine Learning Yearning by Andrew Ng](https://www.goodreads.com/en/book/show/30741739-machine-learning-yearning)

[Calculating the missing value ratio](https://www.analyticsvidhya.com/blog/2021/04/beginners-guide-to-missing-value-ratio-and-its-implementation/)

[Binary classification project with similar dataset](https://www.kaggle.com/code/avi111297/eda-on-employee-promo-pred-got-93-6-accuracy)

[Binary classification project with similar dataset](https://www.kaggle.com/code/amulya9/predict-employee-promotion-result/notebook)

## Data Collection
Used historical dataset with 13 columns:

|Column name           |Variable type|
| -------------        |:-----------:|                       
|employee_id           |Numerical  |
|department            |Categorical |
|region                |Categorical|
|education             |Categorical |
|gender                |Categorical |
|recruitment_channel   |Categorical|
|no_of_trainings       |Numerical |
|age                   |Numerical|
|previous_year_rating  |Numerical|
|length_of_service     |Numerical|
|awards_won?           |Categorical|
|avg_training_score    |Numerical|
|is_promoted           |Categorical|

## Data Cleaning
After pulling the data, I cleaned up the both datasets (historical and current) to reduce noise in the datasets. The changes were made follows:

* Removed duplicates if there are any based on the "employee_id" column, 
* Checked null values and their ratio, and none of the variables are removed since their ratio is quite small,

![ratio_of_missing_values](https://user-images.githubusercontent.com/45776621/163044237-07e1e921-c30b-4775-bf84-01408cd17965.png)

* Filled missing values of 'previous_year_rating' with mean based on 'awards_won?', 'education' and 'recruitment_channel' with most frequent value based on 'department', and 'gender' with mode based on 'awards_won? in the historical dataset,
* #Filled missing values of 'previous_year_rating' on the current dataset with mean based on 'awards_won?' from the historical dataset, and 'education' on the current dataset with most frequent value based on 'department' from historical dataset,
* Replaced Bachelors with Bachelor's in 'education' column for consistency,

**Before:**

![Before](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/education_col_before.png)

**After:**

![After](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/education_col_after.png)

* Replaced FEMALE, Female, and female variables with f and MALE, Male, and male with m in 'gender' column for consistency,

**Before:**

![Before](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/gender_col_before.png)

**After:**

![After](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/gender_col_after.png)

## Exploratory Data Analysis
Visualized the cleaned data to see the trends.

* Created Donut chart for is_promoted data. It looks like the data is imbalanced and needs to be balanced.
![Donut_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/donut_chart.png)

* Created pie charts and stacked bar chart for categorical variables:
**department** variable:
![Pie_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/department_piechart.png)
![Percentage](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/department_percentage.png)

**recruitment_channel** variable:
![Pie_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/recruitment_channel_piechart.png)
![Percentage](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/recruitment_channel_percentage.png)

**education** variable:
![Pie_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/department_piechart.png)
![Percentage](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/education_dist.png)

**gender** variable:
![Pie_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/gender_piechart.png)
![Percentage](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/gender_dist.png)

* Created bar graphs and stacked bar chart for numerical variables:

**age** variable:
![Bar_chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/age.png)
![Dist](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/age_dist.png)

**previous_year_rating** variable:
![Bar_chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/previous_year_ratings.png)
![Dist](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/previous_year_ratings_dist.png)

## Feature Engineering

Categorical variables are encoded, numerical ones are normalized, and 'employee_id' variable is removed from both datasets.

Data were balanced by applying SMOTE, and visualized by the donut chart:

![Donut_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/donut_chart_after_resampling.png)

## Model Building
Data were split into **train (80%)** and **test (20%)** sets.

I used six models *(Decision Tree Classifier, Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Bayes, and KNeighborsClassifier)* to predict the sentiment and evaluated them by using *Accuracy*.

## Model Performance Evalution

I used six models *(Decision Tree Classifier, Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Bayes, and KNeighborsClassifier)* to predict the sentiment and evaluated them by using *Accuracy*.

Random Forest Classifier model performed better than any other models in this project but after tuning the parameters the accuracy dropped to 78% so that's why I used Decision Tree model as it is the second-highest score.

|Model                      |Cross Validation Accuracy Score|                      
| -------------             |:-----------------:|                       
|Decision Tree              |0.9195253234957474|
|Logistic Regression        |0.7659303362220304|
|Support Vector Classifier  |0.8381531307413898|
|Random Forest Classifier   |0.9524331256811436|
|Naive Bayes                |0.6817535632799406|
|K-Neighbots                |0.8508799620065138|

## Hyperparameter Tuning
I got the best accuracy **88.43%** with GridSearchCV and find the optimal hyperparameters.

## Best Model
Applied Decision Tree model with the optimal hyperparameters and got **89.49%** Test Accuracy score.

## Feature Importances
'previous_year_rating', 'avg_training_score', and 'length_of_service' features mostly drives for the promotion. 

![](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/feature_importances.png)

## Predictions for current data

When we apply our model to current dataset, we can expect 13902 current employees get promotion while 9588 employees do not get, the donut chart shows the distribution of getting promoted.

![Donut_Chart](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/donut_chart_pred.png)

## Confusion Matrix
The Confusion Matrix above shows that our model needs to be improved to predict promotions better.

![alt text](https://github.com/cerenkasap/prediction_of_employee_promotion/blob/main/images/confusion_matrix.png "Confusion Matrix of Prediction of Employee promotion")

We estimate the bias as **8.05%** and variance as **1.46%** (10.51-8.05). This classifier is fitting the training set poorly with 8.05% error, but its error on the test set is barely higher than the training error.

The classifier therefore has **high bias,** but **low variance.**

We can say that the algorithm is **underfitting.**

|Data     |Accuracy Score (%)  |Error (%)|                    
| --------|:----------------   |-------:| 
|Training | 91.95              |8.05|
|Test     | 89.49              |10.51|

## Bias vs. Varince tradeoff
**Adding input features** might help to reduce bias on the model.

## Notes
This model needs to be pickled so that it can be saved on disk.

**Error Analysis** should be performed to understand the underlying causes of the error (missclassification).

Thanks for reading :) 
