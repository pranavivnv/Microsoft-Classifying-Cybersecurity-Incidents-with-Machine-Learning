# Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import precision_score,f1_score,recall_score,classification_report,accuracy_score
import joblib
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

data_tr1=pd.read_csv("new_train_sample.csv")
data_tr1
list(data_tr1.columns)  # full column view ("/content/new_train_sample.csv")
pd.set_option('display.max_columns', None)
data_tr1.head(1)
data_tr1.shape
data_tr1.info()
## **Handling data**
(data_tr1.isnull().sum()/(len(data_tr1)))*100
threshold= len(data_tr1)* 0.5   # removing 50% above null value
data_tr1=data_tr1.dropna(thresh=threshold,axis=1)
data_tr1.shape
data_tr1.duplicated().sum()
data_tr1.drop_duplicates(inplace=True)
data_tr1.isnull().sum()
data_tr1.info()
data_tr1["IncidentGrade"].fillna(data_tr1["IncidentGrade"].mode()[0],inplace=True) # fill mode values for IncidentGrade
data_tr1.IncidentGrade.unique()
data_tr1['Timestamp']=pd.to_datetime(data_tr1['Timestamp']) # data clean Timestamp
data_tr1['Timestamp'].head()
data_tr1["Day"]=data_tr1["Timestamp"].dt.day
data_tr1["Month"]=data_tr1["Timestamp"].dt.month
data_tr1["Year"]=data_tr1["Timestamp"].dt.year
data_tr1["Hour"]=data_tr1["Timestamp"].dt.hour
data_tr1["Time"]=data_tr1["Timestamp"].dt.time
data_tr1.drop("Timestamp",axis=1,inplace=True)   # doprring timestamp column
print(data_tr1.head())
data_tr1['IncidentGrade'].value_counts()
data_tr1.Category.unique()
data_tr1.isnull().sum()
# **Data** **Visualization**
# Distribution of target variable
sns.countplot(x="IncidentGrade", data=data_tr1)
plt.show()
# Aggregate data
day_incident_grade = data_tr1.groupby(['Day', 'IncidentGrade']).size().unstack()

# Plot
day_incident_grade.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.xlabel('Day of the Month')
plt.ylabel('Count')
plt.title('IncidentGrade Counts Across Days of the Month')
plt.legend(title='IncidentGrade')
plt.grid(True)
plt.show()
# Aggregate data
day_incident_grade = data_tr1.groupby(['Day', 'IncidentGrade']).size().unstack()

# Plot
day_incident_grade.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.xlabel('Day of the Month')
plt.ylabel('Count')
plt.title('IncidentGrade Counts Across Days of the Month')
plt.legend(title='IncidentGrade')
plt.grid(True)
plt.show()
# Aggregate data
grouped_data=data_tr1.groupby(["Category","IncidentGrade"]).size().reset_index(name="count")

plt.figure(figsize=(20,7))
sns.barplot(data=grouped_data,x="Category", y="count", hue = "IncidentGrade")
plt.xticks(rotation=90)
plt.title("Counts of category by Incident Grade")
plt.show()
# Aggregate data
grouped_data=data_tr1.groupby(["EntityType","IncidentGrade"]).size().reset_index(name="count")

plt.figure(figsize=(20,7))
sns.barplot(data=grouped_data,x="EntityType", y="count", hue = "IncidentGrade")
plt.xticks(rotation=90)
plt.title("Counts of EntityType by Incident Grade")
plt.show()

# dropping 70% above correlated columns
data_tr1.drop(["Month","DeviceName","DeviceId","ResourceIdName","Unnamed: 0","OAuthApplicationId","NetworkMessageId"], axis =1 , inplace = True)

data_tr1['State'].fillna(data_tr1['State'].mode()[0], inplace=True)
data_tr1['CountryCode'].fillna(data_tr1['CountryCode'].mode()[0], inplace=True)
data_tr1['City'].fillna(data_tr1['City'].mode()[0], inplace=True)
data_tr1.info()
# **Feature** **selection** **and** **correlation¶**
# selecting numerical col
numeric_df=data_tr1.select_dtypes(include=['number'])

corr_matrix= numeric_df.corr().abs()

# plot the heatmap
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix,annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, linewidths=0.5)
plt.title('correlation Heatmap')
plt.show()
# dropping 70% above correlated columns
data_tr1.drop(["CountryCode","Sha256","ApplicationName","RegistryValueName","AccountSid","AccountObjectId","FolderPath","AccountUpn"], axis =1 , inplace = True)
# dropping 70% above correlated columns
data_tr1.drop(["RegistryKey","RegistryValueData","DetectorId","Id" ], axis =1 , inplace = True)
# dropping 70% above correlated columns
data_tr1.drop(["OSFamily","OSVersion",'State','FileName','AccountName' ], axis =1 , inplace = True)
data_tr1.info()
#**Encoding**
data_tr1.select_dtypes(include=["object"]).columns
Categorical_coltr=['Category', 'IncidentGrade', 'EntityType', 'EvidenceRole', 'Time']
# **Label** **encode**
encoders = {}
for i in Categorical_coltr:
    le = LabelEncoder()
    data_tr1[i]=le.fit_transform(data_tr1[i])
    encoders[i]=le
data_tr1
data_tr1.select_dtypes(include=["int32", "int64","float64"]).columns
data_tr1.to_csv('/content/datatrain_process.csv',index= False)
# **TRAIN** **TEST** **SPLIT**
data_tr1.head()
from imblearn.over_sampling import RandomOverSampler
os =RandomOverSampler(random_state = 94)
val =data_tr1.drop('IncidentGrade', axis =1)
tar = data_tr1['IncidentGrade']
tar.value_counts()
oval,otar = os.fit_resample(val,tar)
otar.value_counts()  # Class distribution after RandomOverSampler:
# **Feature** **selection**
fs = RandomForestClassifier(n_estimators = 800, random_state =79)
fs.fit(oval,otar)
pd.DataFrame({
    "columns": oval.columns,
    "Score": fs.feature_importances_
}).sort_values('Score', ascending =False).head(15)["columns"].to_list()
oval = oval[['OrgId',
 'IncidentId',
 'AlertId',
 'AlertTitle',
 'Day',
 'Category',
 'Time',
 'EntityType',
 'Hour',
 'IpAddress',
 'City',
 'Url',
 'EvidenceRole',
 'ApplicationId',
 'Year']]
# Train Test Split
traindata,testdata, trainlab,testlab = train_test_split(oval,otar, test_size = 0.20, random_state = 43)
traindata.shape
testdata.shape
# **Model** **building**
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import xgboost
dt_model= DecisionTreeClassifier(max_depth=8,random_state=100,min_samples_split=10,min_samples_leaf=5,max_features="sqrt")
dt_model.fit(traindata,trainlab)
def metrics(actual, predicted):
    acc = accuracy_score(actual, predicted) * 100
    prec = precision_score(actual, predicted, average="macro") * 100
    recall = recall_score(actual, predicted, average="macro") * 100
    macro_f1 = f1_score(actual, predicted, average="macro") * 100

    return pd.DataFrame({
        "Metrics": ["Accuracy", "Precision", "Recall", "Macro F1"],
        "Values": [acc, prec, recall, macro_f1]
    }).set_index("Metrics")
dttr_pred =dt_model.predict(traindata)
dtts_pred =dt_model.predict(testdata)
dttrain_metrics=metrics(trainlab,dttr_pred)

dttest_metrics=metrics(testlab,dtts_pred)
pd.DataFrame({
    "Training":dttrain_metrics["Values"],
    "Testing":dttest_metrics["Values"]
}).reset_index()
# RANDOM FOREST
rfc =RandomForestClassifier(criterion='entropy',max_depth=10,n_estimators=200,n_jobs=-1,random_state=100)
rfc.fit(traindata,trainlab)
rftr_pred =rfc.predict(traindata)
rfts_pred =rfc.predict(testdata)
rftrain_metrics=metrics(trainlab,rftr_pred)
rftest_metrics=metrics(testlab,rfts_pred)
pd.DataFrame({
    "Training":rftrain_metrics["Values"],
    "Testing":rftest_metrics["Values"]
}).reset_index()
# **XGBoost**
# XGBoost
xgbmodel = xgboost.XGBClassifier(n_estimators=200,learning_rate=0.1,random_state=100,n_jobs=-1,max_depth=8)
xgbmodel.fit(traindata,trainlab)
xgbtr_pred =xgbmodel.predict(traindata)
xgbts_pred =xgbmodel.predict(testdata)
xgbtrain_metrics = metrics(trainlab,xgbtr_pred)
xgbtest_metrics =metrics(testlab,xgbts_pred)
pd.DataFrame({
    "Training":xgbtrain_metrics["Values"],
    "Testing":xgbtest_metrics["Values"]
}).reset_index()
