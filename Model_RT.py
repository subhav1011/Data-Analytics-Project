#Random forest regressor for rateings on restaurants in bangalore

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#from matplotlib import rcParams

from sklearn.ensemble import RandomForestRegressor
#from sklearn import metrics


#hyperparameters
n_estimators=1         #number of trees in the random forest
n_features= 10          #Most important features to plot
#locality='Jayanagar'   #To check for a particular locality

data = pd.read_csv("zomato.csv")

#data preprocessing

# drop columns
del data['url']
del data['address']
del data['phone']
del data['location']
del data['reviews_list']

del data['name']

data.rename(columns={'approx_cost(for two people)':'cost_for_two','listed_in(city)':'locality', 'listed_in(type)':'category'},inplace=True)


((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)

data.rate = data.rate.replace("NEW",np.nan)
data.dropna(how='any',inplace=True)
((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
data.rate = data.rate.astype(str)
data.rate = data.rate.apply(lambda x:x.replace('/5',''))
data.rate = data.rate.astype(float)

data.cost_for_two = data.cost_for_two.astype(str)
data.cost_for_two = data.cost_for_two.apply(lambda x:x.replace(',',''))
data.cost_for_two = data.cost_for_two.astype(float)


# To find most important feautures for any locality
##set(data.locality)    #set of localities

#data = data[data.rate >=4.0]
#data = data[data.locality == locality]


'''
data_unique = data.drop_duplicates(subset=['locality', 'name'], keep='first')
data_unique = data_unique[['locality', 'rest_type', 'category', 'rate', 'votes', 'online_order', 'cost_for_two','book_table']]
data_unique = data_unique.sample(frac=1).reset_index(drop=True)
sns.pairplot(data_unique, diag_kind='kde', plot_kws={'alpha':0.2})
'''


data_unique=data
data_unique.info()

data_unique = pd.get_dummies(data_unique)


#split dataset
from sklearn.model_selection import train_test_split

X = data_unique.drop(['rate'], axis=1)
Y = data_unique.rate

X_train , X_valid , y_train, y_valid = train_test_split(X, Y, test_size = 0.3, random_state=1)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape



#Training the Model
print('\ntraining...')
m = RandomForestRegressor(random_state=0, n_estimators=n_estimators, n_jobs=-1)
m.fit(X_train, y_train)
res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]
print('train and test accuracy :',res)


#To find optimal number of estimators
'''
estimators=np.array([ 10,  20,  30,  40, 100, 150, 200, 250, 300])
scores = []
for n in pp_estimators1:
    print(n)
    m.set_params(n_estimators=n)
    m.fit(X_train, y_train)
    scores.append(m.score(X_valid, y_valid))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
'''



# Calculate feature importances
importances = m.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

#Feature graph
vals=n_features
plt.figure(figsize=(10,10))
plt.title("Feature Importance")# for high rated restaurants in "+locality)
plt.xlabel("feature")
plt.ylabel("score")
plt.plot(range(vals),importances[indices][:vals], color = "#3399cc")
#plt.xticks(range(vals), names)
plt.show()

#top few important features
print('top',vals,'features : ')
for i in range(vals):
    print(names[i]," : ",importances[indices][i],'\n')

