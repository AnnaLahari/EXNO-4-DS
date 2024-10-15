# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/86de4554-b531-4204-92fc-63e19abc5437)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/8cc547c4-df5e-4ee1-b941-dd166fedc5f4)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/9ff0c1eb-952e-4d70-b5b8-9ad592c28b61)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/b0e4c975-80fc-4b92-b432-dcf6f7c71a01)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/0a6c7f58-c119-47bd-916d-d7cbe71657fd)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/7af20d19-e10a-4700-9f0c-d94558942fdd)

```
data2
```
![image](https://github.com/user-attachments/assets/f0daae54-b656-48c2-a027-a09061180c31)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/05de0a3d-e09d-4082-b066-cef32fe8c4d1)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/40735853-eb25-421f-87ef-a4c6a92dcfc6)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/1530139a-7e92-42e0-9926-f6f4e6d031b1)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/2eaa2ff3-058d-45f0-8eb7-2db074ca611c)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/28a7b3c1-ed33-462a-a080-00eb2f85de1b)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/e640450a-dbf8-4366-b3d8-cf70765570b4)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/2b8c61b4-44fd-4721-bdbc-1934e977b4dc)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/5698b116-613e-4849-8e92-f928ae5cf937)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/7fe15485-49c7-4166-9e12-4f01d6994ede)

```
data.shape
```
![image](https://github.com/user-attachments/assets/fc768cf0-5ecb-4c69-8ee0-93e0fe9c0124)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/c65f4ff5-43b7-4c33-b9b6-553cfdbe8566)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/5af83648-bc7b-492b-a88e-04fde81d96bf)

```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/d5554a6a-1c11-4df6-a1a1-88c4fd1b59ee)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/c7d5ed13-4bba-4a84-a7a6-3b43351059c7)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/10d94881-23aa-4bee-8d63-c02d01ae2942)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
