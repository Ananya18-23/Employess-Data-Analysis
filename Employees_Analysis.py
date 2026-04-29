#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all  necessary librarires 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# LOading dataset 

df=pd.read_csv("Employee_data.csv")


# In[3]:


# calling for study and checking the dataset

df.head()


# In[4]:


# finding shape of data 

df.shape


# In[5]:


# abhi handel nahi keya hai akrna hai 
df.duplicated().sum()


# In[6]:


df = df.drop_duplicates()
print(df.shape)


# In[7]:


df.duplicated().sum()


# In[8]:


# checking the DataType of each and every column 

df.info()

# According to this there are many columns which have string/Object data type, we have to encode for ml model


# In[9]:


# Checking the null values 

df.isnull().sum()

# Insights: Distance from Home and Company Tenure have null values, since both are numerical columns we can fill this by median of the column


# In[10]:


# filling missing value 

df['Distance from Home'].fillna(df['Distance from Home'].median(), inplace=True)
df['Company Tenure (In Months)'].fillna(df['Company Tenure (In Months)'].median(), inplace=True)


# In[11]:


# checking the result 

df.isnull().sum()


# In[12]:


df.describe()


# # Checking the redundant
# Finding the redundant value in each and every column by finding unique values 


# In[13]:


# since we dont need employee id in the dataset so we can drop 

df=df.drop("Employee ID",axis=1)


# In[14]:


df['Education Level'].unique()  # since spelling error we have to replace 


# In[15]:


# Data profiling 

# Replace the wrong spelling with right spelling 

df['Education Level']=df['Education Level'].replace(["Masterâ€™s Degree","Bachelorâ€™s Degree"],["Masters Degree","Bachelors Degree"])


# In[16]:


df['Education Level'].unique() 


# In[17]:


df['Education Level'].value_counts()


# In[18]:


df.Gender.unique()


# In[19]:


df.Gender.value_counts()


# In[20]:


df['Job Role'].unique()


# In[21]:


# applying loop for fast action 

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
    print()

# since ordinal data do change karinge encoding se, is 'Fair' ko change karna padega nahi to ordinal encoder me problem karega


# In[22]:


# Handling redundant values 

df['Work-Life Balance'] = df['Work-Life Balance'].replace('Fair', 'Below Average')
df['Company Reputation'] = df['Company Reputation'].replace('Fair', 'Poor')


# In[23]:


df.head()


# In[24]:


# According to PDF we are doing train test split and then apply EDA on training and validation data 


# # Train Test split 


# In[25]:


x=df.drop("Attrition", axis=1)
y=df["Attrition"]


# In[26]:


x.head()


# In[27]:


y.head()


# In[28]:


# Split the data into train and test 
from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[29]:


x_train.head()


# In[30]:


x_test.head()


# # Univariate


# In[31]:


sns.histplot(x_train['Age'],kde=True)
plt.title('Distribution of Age')
plt.show()


# In[32]:


sns.histplot(x_train['Years at Company'],kde=True)
plt.title('Distribution of Years at Company')
plt.show()


# In[33]:


sns.histplot(x_train['Monthly Income'],kde=True)
plt.title('Distribution of Monthly Income')
plt.show()


# In[34]:


sns.histplot(x_train['Distance from Home'],kde=True)
plt.title('Distribution of Distance from Home')
plt.show()


# In[35]:


sns.histplot(x_train['Number of Promotions'],kde=True)
plt.title('Distribution of Number of Promotions')
plt.show()


# In[36]:


sns.histplot(x_train['Number of Dependents'],kde=True)
plt.title('Distribution of Number of Dependents')
plt.show()


# In[37]:


sns.histplot(x_train['Company Tenure (In Months)'],kde=True)
plt.title('Distribution of Company Tenure (In Months)')
plt.show()


# # Correlation heatmap
   corr = x_train.corr(numeric_only=True)
   x_train.corr(numeric_only=True)


# In[38]:


# check the multicollinearity 

plt.figure(figsize=(12,10))
corr=x_train.corr()
sns.heatmap(corr,annot=True)


# In[39]:


## feature selection based on correlation

x_train.corr()


# In[40]:


train_df = pd.concat([x_train, y_train], axis=1)

sns.countplot(data=train_df, x='Attrition')
plt.title('Class Balance - Training Data')
plt.show()


# In[41]:


# Almost balance hai 


# # Bivariate analysis


# In[42]:


sns.countplot(train_df,x=y_train,hue=y_train)
plt.title("Bar plot of Attrition")


# In[43]:


sns.countplot(train_df,x=y_train,hue='Gender')
plt.title("Bar plot of Attrition on the basis of gender")


# In[44]:


sns.countplot(train_df,x=y_train,hue='Job Role')
plt.title("Bar plot of Attrition on the basis of Job Role ")


# In[45]:


sns.countplot(train_df,x=y_train,hue='Work-Life Balance')
plt.title("Bar plot of Attrition on the basis of Work-Life Balance")


# In[46]:


sns.countplot(train_df,x=y_train,hue='Job Satisfaction')
plt.title("Bar plot of Attrition on the basis of Job Satisfaction ")


# In[47]:


sns.countplot(train_df,x=y_train,hue='Performance Rating')
plt.title("Bar plot of Attrition on the basis of Performance Rating")


# In[48]:


sns.countplot(train_df,x=y_train,hue='Job Level')
plt.title("Bar plot of Attrition on the basis of Job Level")


# In[49]:


sns.countplot(train_df,x=y_train,hue='Remote Work')
plt.title("Bar plot of Attrition on the basis of remote work")


# In[50]:


sns.countplot(train_df,x=y_train,hue='Leadership Opportunities')
plt.title("Bar plot of Attrition on the basis of Leadership opportunities")


# In[51]:


sns.countplot(train_df,x=y_train,hue='Company Size')
plt.title("Bar plot of Attrition on the basis of Company Size")


# In[52]:


sns.countplot(train_df,x=y_train,hue='Innovation Opportunities')
plt.title("Bar plot of Attrition on the basis of Innovation Opportunities")


# In[53]:


sns.countplot(train_df,x=y_train,hue='Company Reputation')
plt.title("Bar plot of Attrition on the basis of Company Reputation")


# In[54]:


sns.countplot(train_df,x=y_train,hue='Employee Recognition')
plt.title("Bar plot of Attrition on the basis of Employee Recognition")


# In[55]:


sns.countplot(train_df,x=y_train,hue='Education Level')
plt.title("Bar plot of Attrition on the basis of Education Level")


# In[56]:


sns.countplot(train_df,x=y_train,hue='Marital Status')
plt.title("Bar plot of Attrition on the basis of Marital Status")


# # Feature Engineering


# In[57]:


# Ordinal encoding for columns in which order matters 

from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
OE


# In[58]:


Work_Life_Balance = ['Poor', 'Below Average', 'Good', 'Excellent']
Job_Satisfaction = ['Low', 'Medium', 'High', 'Very High']
Performance_Rating = ['Low', 'Below Average', 'Average', 'High']
Education_Level = ['High School', 'Associate Degree', 'Bachelors Degree', 'Masters Degree', 'PhD']
Job_Level = ['Entry', 'Mid', 'Senior']
Company_Size = ['Small', 'Medium', 'Large']
Company_Reputation = ['Poor', 'Good', 'Excellent']
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
x_test[cols] = encoder.transform(x_test[cols])


# In[59]:


x_train.head()


# In[60]:


x_test.head()


# In[61]:


# Label encoding for columns in which order does not matter 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le


# In[62]:


x_train['Gender'] = le.fit_transform(x_train['Gender'])
x_test['Gender'] = le.fit_transform(x_test['Gender'])

x_train['Overtime'] = le.fit_transform(x_train['Overtime'])
x_test['Overtime'] = le.fit_transform(x_test['Overtime'])

x_train['Remote Work'] = le.fit_transform(x_train['Remote Work'])
x_test['Remote Work'] = le.fit_transform(x_test['Remote Work'])

x_train['Leadership Opportunities'] = le.fit_transform(x_train['Leadership Opportunities'])
x_test['Leadership Opportunities'] = le.fit_transform(x_test['Leadership Opportunities'])

x_train['Innovation Opportunities'] = le.fit_transform(x_train['Innovation Opportunities'])
x_test['Innovation Opportunities'] = le.fit_transform(x_test['Innovation Opportunities'])


# In[63]:


x_train.head()


# In[64]:


x_test.head()


# In[65]:


y_train.head()


# In[66]:


x_train = pd.get_dummies(x_train, columns=['Job Role', 'Marital Status'], drop_first=True)
x_test = pd.get_dummies(x_test, columns=['Job Role', 'Marital Status'], drop_first=True)

x_test = x_test.reindex(columns=x_train.columns, fill_value=0)


# In[67]:


print(x_train.select_dtypes(include='object').columns)


# In[68]:


print(x_train.dtypes)


# In[69]:


print(x_test.dtypes)


# In[70]:


# Feature Scaling 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = ['Age', 'Years at Company', 'Monthly Income', 'Distance from Home', 'Number of Promotions','Number of Dependents','Company Tenure (In Months)']

x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])


# In[71]:


print(y_train.dtypes)


# In[72]:


y_train = y_train.map({'Stayed': 1, 'Left': 0})
y_test = y_test.map({'Stayed': 1, 'Left': 0})


# In[73]:


print(y_train.dtypes)


# In[74]:


x_train[num_cols]


# # Model Building 


# In[75]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200)


# In[76]:


x_test.shape


# In[77]:


y_test.shape


# In[78]:


x_train.shape


# In[79]:


y_train.shape


# In[80]:


# this code is from chatgpt

from sklearn.feature_selection import RFE

rfe = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=15)
rfe.fit(x_train, y_train)

# Selected features
selected = pd.Series(rfe.support_, index=x_train.columns)
print(selected[selected == True])


# In[81]:


# Selected features ka list
selected_cols = selected[selected == True].index.tolist()

# Train aur test ko filter karo
x_train_rfe = x_train[selected_cols]
x_test_rfe = x_test[selected_cols]

# Model fit karo
lr = LogisticRegression(max_iter=200)
lr.fit(x_train_rfe, y_train)


# In[82]:


# M ke value 

lr.coef_


# In[83]:


# c ke value 

lr.intercept_


# In[84]:


y_pred=lr.predict(x_test_rfe)


# In[85]:


y_pred


# In[86]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[87]:


print(confusion_matrix(y_test,y_pred))


# In[88]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)


# In[89]:


print(accuracy_score(y_test,y_pred))


# In[90]:


print(classification_report(y_test,y_pred))


# In[91]:


sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Precision:', precision)
print('Recall:', recall)


# In[92]:


from sklearn.metrics import roc_curve, roc_auc_score

y_pred_prob = lr.predict_proba(x_test_rfe)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print('AUC Score:', roc_auc_score(y_test, y_pred_prob))


# In[93]:


# Sensitivity - Specificity tradeoff
sensitivity = tpr
specificity = 1 - fpr

# Jahan dono ka difference minimum ho wahi optimal cutoff
optimal_idx = np.argmin(np.abs(sensitivity - specificity))
optimal_cutoff = thresholds[optimal_idx]
print('Optimal Cutoff:', optimal_cutoff)


# In[94]:


y_pred_optimal = (y_pred_prob >= optimal_cutoff).astype(int)
print(confusion_matrix(y_test, y_pred_optimal))
print(accuracy_score(y_test, y_pred_optimal))


# In[95]:


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
# 3. Recommendation: HR should implement a "Work-Life Balance Audit" for Single employees in Mid-level roles, as they show the highest probability of leaving.


# In[ ]:
