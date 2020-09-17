import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


sns.set(style = 'whitegrid')

data_train = pd.read_csv('../input/healthcare/train_data.csv')
data_test = pd.read_csv('../input/healthcare/test_data.csv')
data_train_dict = pd.read_csv('../input/healthcare/train_data_dictionary.csv')



train = data_train.copy()


test = data_test.copy()


print(data_train.info())
print('\n')
print(data_test.info())



train['Lower_Bound_Age'] = train['Age'].str.split('-', expand = True)[0].astype(int)
train['Upper_Bound_Age'] = train['Age'].str.split('-', expand = True)[1].astype(int)

test['Lower_Bound_Age'] = test['Age'].str.split('-', expand = True)[0].astype(int)
test['Upper_Bound_Age'] = test['Age'].str.split('-', expand = True)[1].astype(int)


num_data = train[['Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient'
             , 'Admission_Deposit', 'Lower_Bound_Age', 'Upper_Bound_Age']]

cat_data = train[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code'
             , 'Department', 'Ward_Type', 'Ward_Facility_Code', 'City_Code_Patient', 'Type of Admission'
             , 'Severity of Illness', 'Stay']]

print(num_data.info())
print('\n')
print(cat_data.info())


fig, ax =plt.subplots(3,2, figsize=(14,10))
fig.tight_layout(pad=5.0)

for ax, n in zip(ax.flatten(), num_data.columns.tolist()):
    sns.distplot(ax=ax, a=num_data[n].dropna(), label="Skewness : %.2f"%(num_data[n].skew()))
    ax.set_title(n, fontsize = 14)
    ax.legend(loc = 'best')


heatmapdata = train[['Stay', 'Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient'
             , 'Admission_Deposit', 'Lower_Bound_Age', 'Upper_Bound_Age', 'City_Code_Patient']]

cormat = heatmapdata.corr()
fig, ax = plt.subplots(figsize = (8,4))
sns.heatmap(data = cormat)
# plt.show()


fig, ax = plt.subplots(cat_data.shape[1],1, figsize = (14, 32))
fig.tight_layout(pad = 5.0)

for ax, n in zip(ax.flatten(), cat_data.columns.tolist()):
    x_axis = cat_data[n].fillna('NaN').value_counts().index
    y_axis = cat_data[n].fillna('NaN').value_counts()
    sns.barplot(ax = ax, x = x_axis, y = y_axis, order =  x_axis)
    ax.set_title(n, fontsize = 14)
    
# plt.show()



train = train[['Stay'] + [col for col in train.columns.tolist() if col != 'Stay']]
test.insert(0, 'Stay', 'NaN')


print('Total null value on data train (%) :\n', np.round(train.isnull().sum() * 100 / len(train), 4))
print('\n')
print('Total null value on data test (%) :\n', np.round(test.isnull().sum() * 100 / len(test), 4))

train.dropna(subset = ['Bed Grade', 'City_Code_Patient'], inplace = True)

test['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace = True)
test['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace = True)

print('Total null value on data train (%) :\n', np.round(train.isnull().sum() * 100 / len(train), 4))
print('\n')
print('Total null value on data test (%) :\n', np.round(test.isnull().sum() * 100 / len(test), 4))


fig, ax = plt.subplots(2,2, figsize = (16,8))
sns.boxplot(ax = ax[0, 0], x = train['Available Extra Rooms in Hospital'])
sns.boxplot(ax = ax[0, 1], x = train['Visitors with Patient'])
sns.boxplot(ax = ax[1, 0], x = train['Admission_Deposit'])
fig.delaxes(ax[1,1])

# plt.show()



q1 = train['Available Extra Rooms in Hospital'].quantile(0.25)
q3 = train['Available Extra Rooms in Hospital'].quantile(0.75)
iqr = q3-q1
train = train[~((train['Available Extra Rooms in Hospital'] < (q1 - 1.5 * iqr)) | (train['Available Extra Rooms in Hospital'] > (q3+1.5*iqr)))]

q1=train['Visitors with Patient'].quantile(0.25)
q3 = train['Visitors with Patient'].quantile(0.75)
iqr = q3-q1
train = train[~ ((train['Visitors with Patient'] < q1 - 1.5 * iqr) | (train['Visitors with Patient'] > (q3 + 1.5 * iqr)))]

q1=train['Admission_Deposit'].quantile(0.25)
q3 = train['Admission_Deposit'].quantile(0.75)
iqr = q3-q1
train = train[~ ((train['Admission_Deposit'] < q1 - 1.5 * iqr) | (train['Admission_Deposit'] > (q3 + 1.5 * iqr)))]

train['Available Extra Rooms in Hospital'] = np.log(train['Available Extra Rooms in Hospital'] + 1)
train['Visitors with Patient'] = np.log(train['Visitors with Patient'] + 1)
train = train[train['Available Extra Rooms in Hospital'] > 0]
train = train[train['Visitors with Patient'] > 0]

test['Available Extra Rooms in Hospital'] = np.log(test['Available Extra Rooms in Hospital'] + 1)
test['Visitors with Patient'] = np.log(test['Visitors with Patient'] + 1)

fig, ax = plt.subplots(2,2, figsize = (16,8))
sns.boxplot(ax = ax[0, 0], x = train['Available Extra Rooms in Hospital'])
sns.boxplot(ax = ax[0, 1], x = train['Visitors with Patient'])
sns.boxplot(ax = ax[1, 0], x = train['Admission_Deposit'])
fig.delaxes(ax[1,1])
# plt.show()


fig, ax =plt.subplots(2,2, figsize=(16,8))
fig.tight_layout(pad=5.0)

sns.distplot(ax=ax[0, 0], a=train['Available Extra Rooms in Hospital']
             , label="Skewness : %.2f"%(train['Available Extra Rooms in Hospital'].skew()))
ax[0, 0].set_title('Available Extra Rooms in Hospital', fontsize = 14)
ax[0, 0].legend(loc = 'best')

sns.distplot(ax=ax[0, 1], a=train['Visitors with Patient']
             , label="Skewness : %.2f"%(train['Visitors with Patient'].skew()))
ax[0, 1].set_title('Visitors with Patient', fontsize = 14)
ax[0, 1].legend(loc = 'best')

sns.distplot(ax=ax[1, 0], a=train['Admission_Deposit']
             , label="Skewness : %.2f"%(train['Admission_Deposit'].skew()))
ax[1, 0].set_title('Admission_Deposit', fontsize = 14)
ax[1, 0].legend(loc = 'best')

fig.delaxes(ax[1,1])

# plt.show()



admission_encode = {'Trauma' : 1, 'Urgent' : 2, 'Emergency' : 3}
train['Type of Admission'] = train['Type of Admission'].map(admission_encode)
test['Type of Admission'] = test['Type of Admission'].map(admission_encode)


severity_encode = {'Minor' : 1, 'Moderate' : 2, 'Extreme' : 3}
train['Severity of Illness'] = train['Severity of Illness'].map(severity_encode)
test['Severity of Illness'] = test['Severity of Illness'].map(severity_encode)

stay_encode = {'0-10' : 1, '11-20' : 2, '21-30' : 3, '31-40' : 4, '41-50' : 5, '51-60' : 6, '61-70' : 7
            ,'71-80' : 8, '81-90' : 9, '91-100' : 10, 'More than 100 Days' : 11}
train['Stay'] = train['Stay'].map(stay_encode)


nominal_data = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code']
testja = pd.DataFrame()
for n in nominal_data:
    ohe = OneHotEncoder(sparse = False, drop = 'first', categories = 'auto')
    ohe.fit(train[nominal_data])
    ohecategory_train = ohe.transform(train[nominal_data])
    ohecategory_test = ohe.transform(test[nominal_data])

    for i in range(ohecategory_train.shape[1]):
        train['dummy_variable_' + n + '_' + str(i)] = ohecategory_train[:,i]
        
    for i in range(ohecategory_test.shape[1]):
        test['dummy_variable_' + n + '_' + str(i)] = ohecategory_test[:,i]


print('Train shape :', train.shape)
print('Test shape :', test.shape)


sc = StandardScaler()
num_col = ['Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient'
             , 'Admission_Deposit', 'Lower_Bound_Age', 'Upper_Bound_Age']
sc.fit(train[num_col])
train[num_col] = sc.transform(train[num_col])
test[num_col] = sc.transform(test[num_col])



print('Train columns :\n',train.columns)
print('Train shape : ', train.shape)
print('\n')
print('Test columns :\n',test.columns)
print('Test shape : ', test.shape)


train.drop(['case_id', 'Hospital_code', 'patientid', 'Age', 'City_Code_Hospital', 'City_Code_Patient'
            , 'Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code']
           , axis = 1, inplace = True)

test.drop(['case_id', 'Hospital_code', 'patientid', 'Age', 'City_Code_Hospital', 'City_Code_Patient'
            , 'Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code']
           , axis = 1, inplace = True)



print('Train columns :\n',train.columns)
print('Train shape : ', train.shape)
print('\n')
print('Test columns :\n',test.columns)
print('Test shape : ', test.shape)

X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values



x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0)
clf = RandomForestClassifier(n_estimators=300, max_depth = 20, min_samples_leaf= 10, max_features=0.5)
clf.fit(x_train_split, y_train_split)
y_pred = clf.predict(x_val_split)
accuracy = accuracy_score(y_pred, y_val_split)
print('Accuracy :',accuracy)


feature_importances = pd.DataFrame( data = {'Features' : train.iloc[:, 1:].columns
                                    ,'Features Importances' : clf.feature_importances_.tolist()})

features = []
fi = []
for n in nominal_data:
    features.append(n)
    fi.append(feature_importances.loc[feature_importances['Features'].str.contains(n), 'Features Importances'].sum())
    
    feature_importances =  feature_importances[~feature_importances['Features'].str.contains(n)]

fi_nominal_data = pd.DataFrame(list(zip(features, fi)), columns = ['Features', 'Features Importances'])
feature_importances = feature_importances.append(fi_nominal_data).sort_values('Features Importances'
                                                                              , ascending = False).reset_index(drop = True)

fig, ax = plt.subplots(figsize=(16,8))
ax = sns.barplot(ax = ax, data = feature_importances.nlargest(20,'Features Importances')
                 ,x='Features Importances',y='Features')
# plt.show()


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


submission = pd.DataFrame()
submission['case_id'] = data_test['case_id']
submission['Stay'] = y_pred

stay_decode = { 1 : '0-10', 2 : '11-20', 3 : '21-30', 4 : '31-40', 5 : '41-50', 6 : '51-60', 7 : '61-70'
            ,8 : '71-80', 9 : '81-90', 10 : '91-100', 11 : 'More than 100 Days'}

submission['Stay'] = submission['Stay'].map(stay_decode)
submission

submission.to_csv(r'Submission.csv', index = False, header = True)

