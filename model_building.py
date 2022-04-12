#import libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold


#pull the datasets
df=pd.read_csv(r'data_scaled.csv', index_col=False)
df_test=pd.read_csv(r'test_scaled.csv', index_col=False)
                   
#Model selection and building

#Model Selection
#Split the dataset
X=df.loc[:, df.columns != 'is_promoted']
y=df['is_promoted']
X_test=df_test

#Resampling for imbalanced data
X_resample, y_resample  = SMOTE().fit_resample(X, y)

#Donut chart
colors = ['#4F6272', '#DD7596']
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
y_resample.value_counts().head(3).plot(kind='pie', labels=None, autopct='%.2f', ax=ax1, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, colors=colors).legend(labels={
                     "1",
                     "0"})
central_circle = plt.Circle((0, 0), 0.4, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title('% of promotions', size=15)
plt.tight_layout()
plt.savefig('images/donut_chart_after_resampling.png', dpi=300)
plt.show()

#Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X_resample, y_resample, test_size=0.2, random_state=42)

#Model building
dt = DecisionTreeClassifier()
lr = LogisticRegression()
svc = SVC()
rf = RandomForestClassifier()
Bayes = BernoulliNB()
KNN = KNeighborsClassifier()

Models = [dt, lr, svc, rf, Bayes, KNN]
Models_Dict = {0: "Decision Tree", 1: "Logistic Regression", 2: "SVC", 3: "Random Forest", 4: "Naive Bayes", 5: "K-Neighbors"}

for i, model in enumerate(Models):
  print("{} Accuracy: {}".format(Models_Dict[i], cross_val_score(model, X_train, y_train, cv = 10, scoring = "accuracy").mean()))

#Hyperparamater Tuning

