
#### MACHINE LEARNING EXERCISE: CLASSIFICATION
# PIMA INDIANS DIABETES

#### Models
* Logistic Regression
* Naive Bayes
* Random Forest Classifier

#### About
* This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

#### Target Variable
* Outcome: 0 (no diabetes) or 1 (has diabetes)

#### Features
1. Pregnancies - Number of times pregnant
1. Glucose - Plasma glucose concentration
1. BloodPressure - Diastolic blood pressure (mm Hg)
1. SkinThickness - Triceps skin fold thickness (mm)
1. Insulin - 2-Hour serum insulin (mu U/ml)
1. BMI - Body mass index
1. DiabetesPedigreeFunction
1. Age

#### Source
* https://www.kaggle.com/uciml/pima-indians-diabetes-database


```python
##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("poster")

%matplotlib inline
```


```python
##### Other Libraries #####

## ML Algorithms ##
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

## For building models ##
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## For measuring performance ##
from sklearn import metrics
from sklearn.model_selection import cross_val_score

## Ignore warnings ##
import warnings
warnings.filterwarnings('ignore')
```

## Load the Dataset


```python
### Load the data
df = pd.read_csv("diabetes.csv")

### Check if the data is properly loaded
print("Size of the dataset:", df.shape)
df.head()
```

    Size of the dataset: (768, 9)
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>




The shape of the loaded dataset is same as what is specified by the source, so we're good to go!

For further inspection, shown below is the list of columns of the data along with its count and type.


```python
### List the columns along with its type
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
    Pregnancies                 768 non-null int64
    Glucose                     768 non-null int64
    BloodPressure               768 non-null int64
    SkinThickness               768 non-null int64
    Insulin                     768 non-null int64
    BMI                         768 non-null float64
    DiabetesPedigreeFunction    768 non-null float64
    Age                         768 non-null int64
    Outcome                     768 non-null int64
    dtypes: float64(2), int64(7)
    memory usage: 54.0 KB
    

## Explore the Dataset


```python
### Summary of statistics
df.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>




The table above summarizes the common statistics stuff we can compute from the data.

By observing the row `count`, we can confirm that there is no missing data.

But if we look at the `min` row, there are columns with zero values that are not expected to have zero values. These columns are `Glucose`, `BloodPressure`, `SkinThickness` and `BMI`. Imagine meeting someone having zero of these attributes, it means that person is a ghost.


```python
### Display the number of zero values per columns 
print("---Count zero values per column---")

for col in ["Glucose", "BloodPressure", "SkinThickness", "BMI"]:
    print("{}: {}".format( col, df[col].value_counts()[0] ))
    
    
### Print the percentage of rows with zero values
print("\n---Rows with zero values in %---")

print("% of rows with zero values in all columns listed above:", 
      (df[(df["Glucose"]==0) | (df["BloodPressure"]==0) | 
          (df["BMI"]==0) | (df["SkinThickness"]==0)].shape[0] / df.shape[0]) * 100)

print("% of rows with zero values in columns 'Glucose', 'BloodPressure' and 'BMI':", 
      (df[(df["Glucose"]==0) | (df["BloodPressure"]==0) | 
          (df["BMI"]==0)].shape[0] / df.shape[0]) * 100)
```

    ---Count zero values per column---
    Glucose: 5
    BloodPressure: 35
    SkinThickness: 227
    BMI: 11
    
    ---Rows with zero values in %---
    % of rows with zero values in all columns listed above: 30.729166666666668
    % of rows with zero values in columns 'Glucose', 'BloodPressure' and 'BMI': 5.729166666666666
    

We can remove rows with zero values in columns `Glucose`, `BloodPressure` or `BMI` since these rows are just around 6% of the data. While, we can impute values for `SkinThickness` because we don't want 30% of our data to be thrown away.

### Relationships

Now, let's look at the correlation between the predictors.


```python
### Determine correlation between variables
df.corr()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pregnancies</th>
      <td>1.000000</td>
      <td>0.129459</td>
      <td>0.141282</td>
      <td>-0.081672</td>
      <td>-0.073535</td>
      <td>0.017683</td>
      <td>-0.033523</td>
      <td>0.544341</td>
      <td>0.221898</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>0.129459</td>
      <td>1.000000</td>
      <td>0.152590</td>
      <td>0.057328</td>
      <td>0.331357</td>
      <td>0.221071</td>
      <td>0.137337</td>
      <td>0.263514</td>
      <td>0.466581</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>0.141282</td>
      <td>0.152590</td>
      <td>1.000000</td>
      <td>0.207371</td>
      <td>0.088933</td>
      <td>0.281805</td>
      <td>0.041265</td>
      <td>0.239528</td>
      <td>0.065068</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>-0.081672</td>
      <td>0.057328</td>
      <td>0.207371</td>
      <td>1.000000</td>
      <td>0.436783</td>
      <td>0.392573</td>
      <td>0.183928</td>
      <td>-0.113970</td>
      <td>0.074752</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>-0.073535</td>
      <td>0.331357</td>
      <td>0.088933</td>
      <td>0.436783</td>
      <td>1.000000</td>
      <td>0.197859</td>
      <td>0.185071</td>
      <td>-0.042163</td>
      <td>0.130548</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0.017683</td>
      <td>0.221071</td>
      <td>0.281805</td>
      <td>0.392573</td>
      <td>0.197859</td>
      <td>1.000000</td>
      <td>0.140647</td>
      <td>0.036242</td>
      <td>0.292695</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>-0.033523</td>
      <td>0.137337</td>
      <td>0.041265</td>
      <td>0.183928</td>
      <td>0.185071</td>
      <td>0.140647</td>
      <td>1.000000</td>
      <td>0.033561</td>
      <td>0.173844</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.544341</td>
      <td>0.263514</td>
      <td>0.239528</td>
      <td>-0.113970</td>
      <td>-0.042163</td>
      <td>0.036242</td>
      <td>0.033561</td>
      <td>1.000000</td>
      <td>0.238356</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>0.221898</td>
      <td>0.466581</td>
      <td>0.065068</td>
      <td>0.074752</td>
      <td>0.130548</td>
      <td>0.292695</td>
      <td>0.173844</td>
      <td>0.238356</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>





```python
### Visualize Correlation

## Generate a mask for the upper triangle
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

## Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

## Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

## Draw the heatmap with the correct aspect ratio
sns.heatmap(df.corr(), mask=mask, cmap=cmap, vmax=.9, square=True, linewidths=.5, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x9797bf0>




![png](Images/output_14_1.png)


`Glucose` has the highest correlation with our target variable `Outcome`, followed by `BMI`. While, `BloodPressure` and `SkinThickness` has the lowest correlation. 

We can look more closely on the relationship of `Outcome` with the predictors using histograms, as shown below. The first histogram of the cell denotes when `Outcome==0 or non-diabetic` while the other one represents when `Outcome==1 or diabetic`. 


```python
### Function to plot histogram
def histplt(col):
    print("----- Outcome vs {}-----".format(col))
    print(df[["Outcome", col]].groupby("Outcome").hist(figsize=(10,3)))
    
### Plot histogram for Outcome vs Pregnancies
histplt("Pregnancies")
```

    ----- Outcome vs Pregnancies-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_16_1.png)



![png](Images/output_16_2.png)


The trend on the `Pregnancies` for both `Outcomes` seems similar, but if we look closely, the average pregnancies for `Outcome==1` seems higher.


```python
### Plot histogram for Outcome vs Glucose
histplt("Glucose")
```

    ----- Outcome vs Glucose-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_18_1.png)



![png](Images/output_18_2.png)


As expected for `Glucose`, diabetic people has higher levels of it while non-diabetic people has the normal `Glucose` which is around 90-100.


```python
### Plot histogram for Outcome vs BloodPressure
histplt("BloodPressure")
```

    ----- Outcome vs BloodPressure-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_20_1.png)



![png](Images/output_20_2.png)


The trends for`BloodPressure` look the same for diabetic and non-diabetic people.


```python
### Plot histogram for Outcome vs SkinThickness
histplt("SkinThickness")
```

    ----- Outcome vs SkinThickness-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_22_1.png)



![png](Images/output_22_2.png)


The average `SkinThickness` of diabetic people looks slightly higher than non-diabetic people.


```python
### Plot histogram for Outcome vs Insulin
histplt("Insulin")
```

    ----- Outcome vs Insulin-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_24_1.png)



![png](Images/output_24_2.png)


Suprisingly, the `insulin` levels for both outcomes are pretty much the same, except that the range of values of `insulin` for non-diabetic people is smaller.


```python
### Plot histogram for Outcome vs BMI
histplt("BMI")
```

    ----- Outcome vs BMI-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_26_1.png)



![png](Images/output_26_2.png)


Same with `SkinThickness`, the average `BMI` of diabetic people looks slightly higher than non-diabetic people.


```python
### Plot histogram for Outcome vs DiabetesPedigreeFunction
histplt("DiabetesPedigreeFunction")
```

    ----- Outcome vs DiabetesPedigreeFunction-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_28_1.png)



![png](Images/output_28_2.png)


The max and average `DiabetesPedigreeFunction`value of diabetic people is higher than that of the non-diabetics.


```python
### Plot histogram for Outcome vs Age
histplt("Age")
```

    ----- Outcome vs Age-----
    Outcome
    0    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    1    [[AxesSubplot(0.125,0.125;0.775x0.755)]]
    dtype: object
    


![png](Images/output_30_1.png)



![png](Images/output_30_2.png)


Most of the non-diabetic people are in their 20s. The distribution of diabetic people with `Age` within the 20-40 is almost uniform, and there are also many diabetic people aged 50 and above.

Lastly, this data is imbalanced as usual. We can resample, but we will not do it for now. The value counts for each outcome is shown below.  


```python
### Check how balanced / imbalanced the data is
df["Outcome"].value_counts()
```




    0    500
    1    268
    Name: Outcome, dtype: int64



## Prepare the Data for Modelling

### Handling Zero Values
##### Remove rows
As stated above, we can remove rows with zero values in columns `Glucose`, `BloodPressure` and `BMI`.


```python
### Create new dataframe wherein the unwanted rows are not included
df_rem = df[ (df["Glucose"]!=0) & (df["BloodPressure"]!=0) & (df["BMI"]!=0) ]

### Check the new dataframe
print("Size of dataframe:", df_rem.shape)
df_rem.head()
```

    Size of dataframe: (724, 9)
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>





```python
### Check minimum values of the new dataframe
df_rem.describe().loc["min"]
```




    Pregnancies                  0.000
    Glucose                     44.000
    BloodPressure               24.000
    SkinThickness                0.000
    Insulin                      0.000
    BMI                         18.200
    DiabetesPedigreeFunction     0.078
    Age                         21.000
    Outcome                      0.000
    Name: min, dtype: float64



##### Impute
Since there are many rows with zero values in `SkinThickness`, we will use Linear Regression to change those values to non-zeroes.


```python
### Separate rows that have zero value in SkinThickness from the rows that have value > 0
df_impute = df_rem[df_rem["SkinThickness"]!=0]
df_0 = df_rem[df_rem["SkinThickness"]==0]

### Use Linear Regression for imputation
## Instantiate the Linear Regression Algorithm
linreg = LinearRegression()

## Fit the dataframe with SkinThickness > 0 on linreg
linreg.fit(df_impute.drop(["SkinThickness", "Outcome"], axis=1), df_impute["SkinThickness"])

## Get the new values of SkinThickness
df_0["SkinThickness"] = linreg.predict(df_0.drop(["SkinThickness","Outcome"], axis=1))

### Merge the imputed datas, then check
df_impute = df_impute.append(df_0)
df_impute.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
      <td>724.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.866022</td>
      <td>121.882597</td>
      <td>72.400552</td>
      <td>29.024005</td>
      <td>84.494475</td>
      <td>32.467127</td>
      <td>0.474765</td>
      <td>33.350829</td>
      <td>0.343923</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.362803</td>
      <td>30.750030</td>
      <td>12.379870</td>
      <td>9.683955</td>
      <td>117.016513</td>
      <td>6.888941</td>
      <td>0.332315</td>
      <td>11.765393</td>
      <td>0.475344</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>24.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>18.200000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.750000</td>
      <td>64.000000</td>
      <td>22.014105</td>
      <td>0.000000</td>
      <td>27.500000</td>
      <td>0.245000</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>29.000000</td>
      <td>48.000000</td>
      <td>32.400000</td>
      <td>0.379000</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>142.000000</td>
      <td>80.000000</td>
      <td>35.004675</td>
      <td>130.500000</td>
      <td>36.600000</td>
      <td>0.627500</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>




### Train-Test Split


```python
### Seaprate the predictors from the target variable
X = df_impute.drop(["Outcome"], axis=1)
y = df_impute["Outcome"]

print("Size of x (predictors):\t{}\nSize of y (target):\t{}".format(X.shape, y.shape))
```

    Size of x (predictors):	(724, 8)
    Size of y (target):	(724,)
    


```python
### Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

### Check shape to make sure it is all in order
print("Size of x_train: {} \t Size of x_test: {} \nSize of y_train: {} \t Size of y_test: {}".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape))
```

    Size of x_train: (506, 8) 	 Size of x_test: (218, 8) 
    Size of y_train: (506,) 	 Size of y_test: (218,)
    


```python
print(y_train.value_counts(), '\n', y_test.value_counts())
```

    0    332
    1    174
    Name: Outcome, dtype: int64 
     0    143
    1     75
    Name: Outcome, dtype: int64
    

### Standard Scaler


```python
### Instantiate the Standard Scaler
scaler = StandardScaler()

### Fit the scaler to the training set
scaler.fit(X_train)

### Transform the training set
X_train_scaled = scaler.transform(X_train)

### Transform the test set
X_test_scaled = scaler.transform(X_test)
```


```python
### Change to Pandas dataframe for easier viewing and manipulation of the data
X_train_sdf = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_sdf = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
```

## Build the Models


```python
### Initialized for easy plotting of confusion matrix
def confmatrix(y_pred, title):
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    plt.title(title)
    
    sns.set(font_scale=1.4) # For label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) # Font size
```

### Logistic Regression

#### Build/Train the Model


```python
### Instantiate the Algorithm 
logreg = LogisticRegression()

### Train/Fit the model
logreg.fit(X_train_scaled, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



#### Validate the Model


```python
### Predict on the test set
logreg_pred = logreg.predict(X_test_scaled)
```

##### Classification Report


```python
### Get performance metrics
logreg_score = metrics.accuracy_score(y_test, logreg_pred) * 100

### Print classification report
print("Classification report for {}:\n{}".format(logreg, metrics.classification_report(y_test, logreg_pred)))
print("Accuracy score:", logreg_score)
```

    Classification report for LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False):
                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.85       143
               1       0.75      0.63      0.68        75
    
        accuracy                           0.80       218
       macro avg       0.78      0.76      0.77       218
    weighted avg       0.79      0.80      0.79       218
    
    Accuracy score: 79.81651376146789
    

The accuracy score and precision of this model is pretty good. Though, we still need to cross-validate this to know if this is luck or not.

Shown below is the confusion matrix.

##### Confusion Matrix


```python
### Plot the confusion matrix
confmatrix(logreg_pred, "LogReg - Pima Indians Diabetes\nConfusion Matrix")
```


![png](Images/output_57_0.png)


##### Cross-Validation


```python
### Perform 10-fold cross-validation
logreg_cv = np.mean(cross_val_score(logreg, X, y, cv=10) * 100)
print("10-Fold Cross-Validation score for KNN fit in Regular Training Set:", logreg_cv)
```

    10-Fold Cross-Validation score for KNN fit in Regular Training Set: 75.70119729028661
    

The results of cross-validation for logistic regression is also good, which proves that the accuracy score got previously for this model is not pure luck.

### Gaussian Naive Bayes

#### Build/Train the Model


```python
### Instantiate the Algorithm 
gnb = GaussianNB()

### Train the model
gnb.fit(X_train_scaled, y_train)
```




    GaussianNB(priors=None, var_smoothing=1e-09)



#### Validate the Model


```python
### Predict on the Test Set
gnb_pred = gnb.predict(X_test_scaled)
```

##### Classification Report


```python
### Get performance metrics
gnb_score = metrics.accuracy_score(y_test, gnb_pred) * 100

### Print classification report
print("Classification report for {}:\n{}".format(gnb, metrics.classification_report(y_test, gnb_pred)))
print("Accuracy score:", gnb_score)
```

    Classification report for GaussianNB(priors=None, var_smoothing=1e-09):
                  precision    recall  f1-score   support
    
               0       0.82      0.85      0.83       143
               1       0.69      0.65      0.67        75
    
        accuracy                           0.78       218
       macro avg       0.76      0.75      0.75       218
    weighted avg       0.78      0.78      0.78       218
    
    Accuracy score: 77.98165137614679
    

This model also gave good accuracy score. Its recall score is better than that of LogReg.

##### Confusion Matrix


```python
### Plot the confusion matrix
confmatrix(gnb_pred, "GNB - Pima Indians Diabetes\nConfusion Matrix")
```


![png](Images/output_70_0.png)


##### Cross-Validation


```python
### Perform cross-validation then get the mean
gnb_cv = np.mean(cross_val_score(gnb, X, y, cv=10) * 100)
print("10-Fold Cross-Validation score for KNN fit in Regular Training Set:", gnb_cv)
```

    10-Fold Cross-Validation score for KNN fit in Regular Training Set: 75.0046894762793
    

### Random Forest Classifier

#### Build/Train the Model


```python
### Instantiate algorithm
rf = RandomForestClassifier()

### Fit the model to the data
rf.fit(X_train_scaled, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



#### Validate the Model


```python
### Predict on the test set
rf_pred = rf.predict(X_test_scaled)
```

##### Classification Report


```python
### Get performance metrics
rf_score = metrics.accuracy_score(y_test, rf_pred) * 100

### Print classification report
print("Classification report for {}:\n{}".format(rf, metrics.classification_report(y_test, rf_pred)))
print("Accuracy score:", rf_score)
```

    Classification report for RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False):
                  precision    recall  f1-score   support
    
               0       0.78      0.87      0.82       143
               1       0.67      0.52      0.59        75
    
        accuracy                           0.75       218
       macro avg       0.72      0.69      0.70       218
    weighted avg       0.74      0.75      0.74       218
    
    Accuracy score: 74.77064220183486
    

This Random Forest model got a decent accuracy score but not as good as the previous models. This also has the lowest recall score.

##### Confusion Matrix


```python
### Plot the confusion matrix
confmatrix(rf_pred, "RF - Pima Indians Diabetes\nConfusion Matrix")
```


![png](Images/output_82_0.png)


##### Cross-Validation


```python
### Perform cross-validation then get the mean
rf_cv = np.mean(cross_val_score(rf, X, y, cv=10) * 100)
print("10-Fold Cross-Validation score for KNN fit in Regular Training Set:", rf_cv)
```

    10-Fold Cross-Validation score for KNN fit in Regular Training Set: 75.55263468175873
    

## Summary of the Results


```python
df_results = pd.DataFrame.from_dict({
    'Accuracy Score':{'Logistic Regression':logreg_score, 'Gaussian Naive Bayes':gnb_score, 'Random Forest':rf_score},
    'Cross-Validation Score':{'Logistic Regression':logreg_cv, 'Gaussian Naive Bayes':gnb_cv, 'Random Forest':rf_cv}
    })
df_results
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy Score</th>
      <th>Cross-Validation Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gaussian Naive Bayes</th>
      <td>77.981651</td>
      <td>75.004689</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>79.816514</td>
      <td>75.701197</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>74.770642</td>
      <td>75.552635</td>
    </tr>
  </tbody>
</table>




**We got good accuracy scores from all of the models, but not that good precision and recall for classifying people with diabetes.** One factor for this is the imbalance of data since more or less 65% of the subjects in the dataset have no diabetes. Also, there may be other predictors for diabetes that are not included in this dataset.

**Logistic Regression shows more promise.** This model does not only have the highest accuracy score and cross-validation score, it also has good precision of 75%. The f1-score of this model is 68%. 

While, **Naive Bayes has 65% recall which is better compared to the other models.**

Overall, more data and more fine tuning are needed.

## Special Thanks
* [FTW Foundation](https://ftwfoundation.org)
