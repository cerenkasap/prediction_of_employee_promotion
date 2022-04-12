## Prediction of Employee Promotion - Binary Classification 🏆: Project Overview

Created a model that can classify a employee promotion as a Positive or a Negative with **(89% Accuracy)**.

Used two dataset: PromosSample.csv as a traning set with **54808 examples** and test.csv as a test dataset **23490 examples** using pandas library in python.

Applied **Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Naive Bayes**, and **KNeighborsClassifier** and optimized using **GridSearchCV** to find the best model.

### Code Used

Python version: *Python 3.7.11* 

Packages: *pandas, o seaborn, matplotlib, numpy, scikit-learn, and SMOTE*

### Resources Used

[Calculating the missing value ratio](https://www.analyticsvidhya.com/blog/2021/04/beginners-guide-to-missing-value-ratio-and-its-implementation/)

[Binary classification project with similar dataset](https://www.kaggle.com/code/avi111297/eda-on-employee-promo-pred-got-93-6-accuracy)

[Binary classification project with similar dataset](https://www.kaggle.com/code/amulya9/predict-employee-promotion-result/notebook)

## Data Collection
Used training set with 13 columns:

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
After pulling the data, I cleaned up the both dataset (training and test) to reduce noises in the datasets. The changes were made follows:

* Removed duplicates if there is any based on "employee_id" column, 
* Checked null values and their ratio,
* Filled missing values of 'previous_year_rating' with mean based on 'awards_won?', 'education' and 'recruitment_channel' with most frequent value based on 'department', and 'gender' with mode based on 'awards_won? in the training dataset,
* #Filled missing values of 'previous_year_rating' with mean based on 'awards_won?' from training set, and 'education' with most frequent value based on 'department' from training set,
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



## Model Building
![ratio_of_missing_values](https://user-images.githubusercontent.com/45776621/163044237-07e1e921-c30b-4775-bf84-01408cd17965.png)


Data were split into **train (80%)** and **test (20%)** sets.

I used six models *(Decision Tree Classifier, Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Bayes, and KNeighborsClassifier)* to predict the sentiment and evaluated them by using *Accuracy*.

## Model Performance Evalution
Logistic Regression model performed better than any other models in this project.

|Model                      |Test Accuracy Score|                      
| -------------             |:-----------------:|                       
|Decision Tree              |0.9195253234957474|
|Logistic Regression        |0.7659303362220304|
|Support Vector Classifier  |0.8381531307413898|
|Random Forest Classifier   |0.9524331256811436|
|Naive Bayes                |0.6817535632799406|
|K-Neighbots                |0.8508799620065138|

## Hyperparameter Tuning


## Best Model


## Confusion Matrix


Thanks for reading :) 
