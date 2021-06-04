#Exploratory Data Analysis on zomato.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


df1 = pd.read_csv("zomato.csv")
print(df1.head(10))
df1.info()


#Data-Cleaning 

#Dropping Redundant Columns
df2 = df1.drop(['url','phone','dish_liked'],axis = 1)
#df2.info()

#Removing Redundant Rows
df2.duplicated().sum()
df2.drop_duplicates(inplace=True)

#Replacing NaN Values
df2.isnull().sum()
df2.dropna(how='any',inplace=True)
#df2.info()

#Altering some column names
df2.columns
df2 = df2.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})

#Altering DataTypes
comma_rem = lambda x: int(x.replace(',', '')) if type(x) == np.str and x != np.nan else x 
df2.votes = df2.votes.astype('int')
df2['cost'] = df2['cost'].apply(comma_rem)
df2['cost'] = df2['cost'].astype(float)

#Stripping /5 from rates and converting to float
df2 = df2.loc[df2.rate !='NEW']
df2 = df2.loc[df2.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
df2.rate = df2.rate.apply(remove_slash).str.strip().astype('float')
#df2.info()

#Plotting a heatmap between our input variables
corr = df2.corr(method='kendall')
plt.figure(figsize=(15,8))
plt.title("Correlation between rate, votes and cost")
sns.heatmap(corr, annot=True)
df2.columns

#Popular locations
plt.figure(figsize=(7,7))
plt.title("Foodie hotspots")
plt.xlabel('Count')
Rest_locations=df2['location'].value_counts()[:20]
sns.barplot(Rest_locations,Rest_locations.index,palette="rainbow")


#Popular cuisines
plt.figure(figsize=(7,7))
cuisines=df2['cuisines'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index,palette='rainbow')
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")


#Plot for Restaurants Rate and Booktable service
# Building a figure
fig = plt.figure(constrained_layout=True, figsize=(15, 12))

# Axis definition with GridSpec
gs = GridSpec(2, 5, figure=fig)
ax2 = fig.add_subplot(gs[0, :3])
ax3 = fig.add_subplot(gs[0, 3:])

sns.kdeplot(df2.query('rate > 0 & book_table == "Yes"')['rate'], ax=ax2,
             color='blue', shade=True, label='With Book Table Service')
sns.kdeplot(df2.query('rate > 0 & book_table == "No"')['rate'], ax=ax2,
             color='red', shade=True, label='Without Book Table Service')
ax2.set_title('Restaurants Rate Distribution by Book Table Service Offer', color='dimgrey', size=14)
sns.boxplot(x='book_table', y='rate', data=df2, palette=['green', 'red'], ax=ax3)
ax3.set_title('Box Plot for Rate and Book Table Service', color='dimgrey', size=14)


sns.lmplot(x="rate",y="cost", data=df2);

#violin plot for approx cost for two people
plt.figure(figsize=(15,8))
sns.violinplot(df2.cost)
plt.title('Approx cost for 2 people distribution', size = 20, pad = 15)
plt.xlabel('Approx cost for 2 people',size = 15)
plt.ylabel('Density',size = 15)




