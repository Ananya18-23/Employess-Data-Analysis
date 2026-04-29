#!/usr/bin/env python
# coding: utf-8

# import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
df = pd.read_csv("Employee_data.csv")

# calling for study and checking the dataset
df.head()

# finding shape of data
df.shape

# checking duplicates
df.duplicated().sum()

df = df.drop_duplicates()
print(df.shape)

df.duplicated().sum()

# checking the DataType of each and every column
df.info()
# According to this there are many columns which have string/Object data type, we have to encode for ml model

# Checking the null values
df.isnull().sum()
# Insights: Distance from Home and Company Tenure have null values, since both are numerical columns we can fill this by median of the column

# filling missing value
df['Distance from Home'].fillna(df['Distance from Home'].median(), inplace=True)
df['Company Tenure (In Months)'].fillna(df['Company Tenure (In Months)'].median(), inplace=True)

# checking the result
df.isnull().sum()

df.describe()

# Checking the redundant - Finding the redundant value in each and every column by finding unique values

# since we dont need employee id in the dataset so we can drop
df = df.drop("Employee ID", axis=1)

df['Education Level'].unique()  # since spelling error we have to replace

# Data profiling - Replace the wrong spelling with right spelling
df['Education Level'] = df['Education Level'].replace(
    ["Masterâ€™s Degree", "Bachelorâ€™s Degree"],
    ["Masters Degree", "Bachelors Degree"]
)

df['Education Level'].unique()
df['Education Level'].value_counts()
df.Gender.unique()
df.Gender.value_counts()
df['Job Role'].unique()

# applying loop for fast action
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
    print()

# since ordinal data, is 'Fair' ko change karna padega nahi to ordinal encoder me problem karega

# Handling redundant values
df['Work-Life Balance'] = df['Work-Life Balance'].replace('Fair', 'Below Average')
df['Company Reputation'] = df['Company Reputation'].replace('Fair', 'Poor')

df.head()

# According to PDF we are doing train test split and then apply EDA on training and validation data

# ===================== Train Test split =====================

x = df.drop("Attrition", axis=1)
y = df["Attrition"]

x.head()
y.head()

# Split the data into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_train.head()
x_test.head()

# ===================== Univariate =====================

sns.histplot(x_train['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

sns.histplot(x_train['Years at Company'], kde=True)
plt.title('Distribution of Years at Company')
plt.show()

sns.histplot(x_train['Monthly Income'], kde=True)
plt.title('Distribution of Monthly Income')
plt.show()

sns.histplot(x_train['Distance from Home'], kde=True)
plt.title('Distribution of Distance from Home')
plt.show()

sns.histplot(x_train['Number of Promotions'], kde=True)
plt.title('Distribution of Number of Promotions')
plt.show()

sns.histplot(x_train['Number of Dependents'], kde=True)
plt.title('Distribution of Number of Dependents')
plt.show()

sns.histplot(x_train['Company Tenure (In Months)'], kde=True)
plt.title('Distribution of Company Tenure (In Months)')
plt.show()

# ===================== Correlation heatmap =====================

# check the multicollinearity
plt.figure(figsize=(12, 10))
corr = x_train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True)
plt.show()

# feature selection based on correlation
x_train.select_dtypes(include=[np.number]).corr()

train_df = pd.concat([x_train, y_train], axis=1)

sns.countplot(data=train_df, x='Attrition')
plt.title('Class Balance - Training Data')
plt.show()

# Almost balance hai

# ===================== Bivariate analysis =====================

sns.countplot(data=train_df, x='Attrition', hue='Attrition')
plt.title("Bar plot of Attrition")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Gender')
plt.title("Bar plot of Attrition on the basis of gender")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Job Role')
plt.title("Bar plot of Attrition on the basis of Job Role")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Work-Life Balance')
plt.title("Bar plot of Attrition on the basis of Work-Life Balance")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Job Satisfaction')
plt.title("Bar plot of Attrition on the basis of Job Satisfaction")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Performance Rating')
plt.title("Bar plot of Attrition on the basis of Performance Rating")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Job Level')
plt.title("Bar plot of Attrition on the basis of Job Level")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Remote Work')
plt.title("Bar plot of Attrition on the basis of remote work")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Leadership Opportunities')
plt.title("Bar plot of Attrition on the basis of Leadership opportunities")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Company Size')
plt.title("Bar plot of Attrition on the basis of Company Size")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Innovation Opportunities')
plt.title("Bar plot of Attrition on the basis of Innovation Opportunities")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Company Reputation')
plt.title("Bar plot of Attrition on the basis of Company Reputation")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Employee Recognition')
plt.title("Bar plot of Attrition on the basis of Employee Recognition")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Education Level')
plt.title("Bar plot of Attrition on the basis of Education Level")
plt.show()

sns.countplot(data=train_df, x='Attrition', hue='Marital Status')
plt.title("Bar plot of Attrition on the basis of Marital Status")
plt.show()

# ===================== Feature Engineering =====================

# Ordinal encoding for columns in which order matters
from sklearn.preprocessing import OrdinalEncoder

Work_Life_Balance    = ['Poor', 'Below Average', 'Good', 'Excellent']
Job_Satisfaction     = ['Low', 'Medium', 'High', 'Very High']
Performance_Rating   = ['Low', 'Below Average', 'Average', 'High']
Education_Level      = ['High School', 'Associate Degree', 'Bachelors Degree', 'Masters Degree', 'PhD']
Job_Level            = ['Entry', 'Mid', 'Senior']
Company_Size         = ['Small', 'Medium', 'Large']
Company_Reputation   = ['Poor', 'Good', 'Excellent']
Employee_Recognition = ['Low', 'Medium', 'High', 'Very High']

encoder = OrdinalEncoder(categories=[Work_Life_Balance, Job_Satisfaction,
                                     Performance_Rating, Education_Level,
                                     Job_Level, Company_Size,
                                     Company_Reputation, Employee_Recognition])

cols = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
        'Education Level', 'Job Level', 'Company Size',
        'Company Reputation', 'Employee Recognition']

# fit on train only, transform both
x_train[cols] = encoder.fit_transform(x_train[cols])
x_test[cols]  = encoder.transform(x_test[cols])

x_train.head()
x_test.head()

# Label encoding for columns in which order does not matter
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in ['Gender', 'Overtime', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities']:
    x_train[col] = le.fit_transform(x_train[col].astype(str))
    x_test[col]  = le.fit_transform(x_test[col].astype(str))

x_train.head()
x_test.head()
y_train.head()

# One-Hot Encoding for nominal columns
x_train = pd.get_dummies(x_train, columns=['Job Role', 'Marital Status'], drop_first=True)
x_test  = pd.get_dummies(x_test,  columns=['Job Role', 'Marital Status'], drop_first=True)

# Align columns - fill missing with 0
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

# Convert all columns to float to remove NaN / bool issues
x_train = x_train.astype(float)
x_test  = x_test.astype(float)

# Verify no NaN remaining
print("NaN in x_train:", x_train.isnull().sum().sum())
print("NaN in x_test :", x_test.isnull().sum().sum())

print(x_train.select_dtypes(include='object').columns)
print(x_train.dtypes)
print(x_test.dtypes)

# ===================== Feature Scaling =====================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = ['Age', 'Years at Company', 'Monthly Income', 'Distance from Home',
            'Number of Promotions', 'Number of Dependents', 'Company Tenure (In Months)']

x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols]  = scaler.transform(x_test[num_cols])

print(y_train.dtypes)

y_train = y_train.map({'Stayed': 1, 'Left': 0})
y_test  = y_test.map({'Stayed': 1, 'Left': 0})

print(y_train.dtypes)

x_train[num_cols]

# ===================== Model Building =====================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=200)

print("x_test shape :", x_test.shape)
print("y_test shape :", y_test.shape)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

from sklearn.feature_selection import RFE

rfe = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=15)
rfe.fit(x_train, y_train)

selected = pd.Series(rfe.support_, index=x_train.columns)
print(selected[selected == True])

# Selected features ka list
selected_cols = selected[selected == True].index.tolist()

x_train_rfe = x_train[selected_cols]
x_test_rfe  = x_test[selected_cols]

lr = LogisticRegression(max_iter=200)
lr.fit(x_train_rfe, y_train)

print("Coefficients :", lr.coef_)
print("Intercept    :", lr.intercept_)

y_pred = lr.predict(x_test_rfe)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision   = tp / (tp + fp)
recall      = tp / (tp + fn)

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Precision  :', precision)
print('Recall     :', recall)

# ===================== ROC Curve =====================
from sklearn.metrics import roc_curve, roc_auc_score

y_pred_prob = lr.predict_proba(x_test_rfe)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print('AUC Score:', roc_auc_score(y_test, y_pred_prob))

# Sensitivity - Specificity tradeoff
sensitivity = tpr
specificity = 1 - fpr

optimal_idx    = np.argmin(np.abs(sensitivity - specificity))
optimal_cutoff = thresholds[optimal_idx]
print('Optimal Cutoff:', optimal_cutoff)

y_pred_optimal = (y_pred_prob >= optimal_cutoff).astype(int)
print(confusion_matrix(y_test, y_pred_optimal))
print(accuracy_score(y_test, y_pred_optimal))

plt.plot(thresholds, sensitivity, label='Sensitivity')
plt.plot(thresholds, specificity, label='Specificity')
plt.axvline(x=optimal_cutoff, color='red', linestyle='--', label='Optimal Cutoff')
plt.xlabel('Threshold')
plt.title('Sensitivity vs Specificity')
plt.legend()
plt.show()

# Conclusion:
# 1. Job Level (Senior) and Remote Work are the strongest predictors of staying.
# 2. Single Marital Status and High Overtime are the biggest threats to retention.
# 3. Recommendation: HR should implement a "Work-Life Balance Audit" for Single employees
#    in Mid-level roles, as they show the highest probability of leaving.
