# DA_final_project
Submission for a data analytics project for college.

Dataset chosen: Zomato bangalore restaurants
The dataset was too big to upload onto github, so please refer to this link to download the dataset before running the python script
https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants


EDA.py:
File performs exploratory data analysis on the zomato.csv dataset
Plots graphs for:
	1)Heatmap between our input variables
	2)Popular locations
	3)Popular cuisines
	4)Restaurants Rate and Booktable service
	5)Density of approx cost for two people



Model_RT.py
Performs Data preprocessing, Splits data into training and validation
Uses Random Forest Regressor as the machine learning model, which it trains over the train split.
Also outputs the feature importance graph and the scores for the top few most important features

Hyperparameters:
n_estimators: The number of trees in the forest
n_features: The number of top most important features for the RF model
*locality: locality to explore deeper into

*Code allows User to uncomment and edit a few lines and run to gain insight on important features for highly rated restaurants in specific localities.
To edit:
set locality parameter to desired locality
uncomment lines 17, 52,53,118



