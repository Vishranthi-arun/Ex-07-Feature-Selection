# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file.

## Explanation
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

## ALGORITHM
### STEP 1
Read the given Data

### STEP 2
Clean the Data Set using Data Cleaning Process

### STEP 3
Apply Feature selection techniques to all the features of the data set

### STEP 4
Save the data to the file

## CODE
```
Developed by: Vishranthi A
Register number: 212221230124

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS

X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='purple')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.Forward feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.Backward feature elimination

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.Bi-directional elimination

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.Recursive Feature Selection

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.Random Forest Importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="purple")
```
## OUPUT
<img width="502" alt="169986490-74ed3ec2-a7fd-4cfd-96e0-6e72ddddaf10" src="https://user-images.githubusercontent.com/93427278/170059652-ecac492d-58e1-469d-8ed8-80fc6b996a09.png">

Analyzing the boston dataset:
<img width="639" alt="169986857-aeac3bc6-601e-4d1e-8d0a-5b78eb0de61e" src="https://user-images.githubusercontent.com/93427278/170059739-2ca715f7-f9e7-4f73-82a5-a659df046120.png">

<img width="593" alt="169986959-f3e7b192-a374-4c4c-8c82-b7b4feeb775a" src="https://user-images.githubusercontent.com/93427278/170059846-e16f0c4d-9c7e-4c2b-9bdf-8d1263aa4e0f.png">

<img width="572" alt="169986998-587d6168-30ef-409e-b1e8-65679ba04bd3" src="https://user-images.githubusercontent.com/93427278/170060583-ce248897-0175-446b-9631-277287c30310.png">

Analyzing dataset using Distplot:
<img width="658" alt="ds" src="https://user-images.githubusercontent.com/93427278/170062051-3b4b6994-44f1-45f0-8aad-3301ab65ac45.png">

## Filter Methods:
Variance Threshold:
![169987986-f1ab4f59-29d0-40b5-ad42-7f48725a6f7c](https://user-images.githubusercontent.com/93427278/170062195-342db6b6-5ecf-4e19-819e-d97f0b1a7612.png)

Information Gain:
![169988217-b30de1e4-da27-4deb-91c7-d3baa35be0d2](https://user-images.githubusercontent.com/93427278/170062251-9208af20-79b9-42cf-8c3e-62441b1eccec.png)

SelectKBest Model:
![169988326-73b16a8f-62dd-4cc4-9641-4326369c19d0](https://user-images.githubusercontent.com/93427278/170062297-51f0d29a-2612-43fb-8001-d9ce8c9748ec.png)

Correlation Coefficient:
![169988374-40e71b1c-10be-463c-a5e1-f39771600c47](https://user-images.githubusercontent.com/93427278/170062340-9db2c60b-59c0-4398-a1da-aedc03f70348.png)

Mean Absolute difference:
![169988519-41c34a56-afb8-46ed-adc0-11023400a261](https://user-images.githubusercontent.com/93427278/170062388-27505cf5-8eee-4b08-8950-a6bf6b06cb73.png)

Chi Square Test:
![169988578-b6f45ff9-63fc-4863-bd94-86e5633b81c8](https://user-images.githubusercontent.com/93427278/170062449-44c326c2-9e23-4a2a-aa3b-440ac3b3132a.png)
![169988648-31329a08-6498-4700-9b89-218f47edb6a6](https://user-images.githubusercontent.com/93427278/170062483-af539458-30b8-4bf8-9a7c-0c049889ea6d.png)

SelectPercentile Method:
![169988771-7fcd3705-624f-4121-8cf9-a52a3a1e7624](https://user-images.githubusercontent.com/93427278/170062524-e6ee2f3e-8f49-452d-93b2-f55cc6eadbc2.png)

## Wrapper Methods:
Forward Feature Selection:
![169989814-bea05be1-698d-4c20-a253-98e165f4211b](https://user-images.githubusercontent.com/93427278/170062678-726e9650-9403-455a-9f8c-d486f6ae4df2.png)

Backward Feature Selection:
![169990113-baa39ac0-0d2d-4b50-bfba-408c7c333e33](https://user-images.githubusercontent.com/93427278/170062722-6c961a8e-c74a-45e0-9347-acc686c7131f.png)

Bi-Directional Elimination:
![169990149-76d44726-8a4f-40e1-b482-ae0621f27a52](https://user-images.githubusercontent.com/93427278/170062760-45a2e6af-086b-4f97-901e-6d09094f828c.png)

Recursive Feature Selection:
![169990185-62be02f6-e69b-453a-acb4-367ac6470660](https://user-images.githubusercontent.com/93427278/170062794-840668a5-a434-4830-a23c-d87cd76afcc4.png)

## Embedded Methods:
Random Forest Importance:
![169990230-99433abd-e0ed-402f-992a-d86e0de11ef4](https://user-images.githubusercontent.com/93427278/170062874-d3c99008-345f-4e94-9370-1bba686e7440.png)

## RESULT:
Hence various feature selection techniques are applied to the given data set successfully and saved the data into a file.
