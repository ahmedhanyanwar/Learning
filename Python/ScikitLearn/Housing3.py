#### chapter 2 from 72 to end
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree

HOUSING_PATH = os.path.join("datasets","housing") 

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
##  make stratified from median_income attribute gives me test set have same numerical calculation of train set
def makeStratified(housing):
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        strat_train_set = housing.loc[train_index] 
        strat_test_set =  housing.loc[test_index]
    return strat_train_set,strat_test_set

def displayScores(scores):
    print("Scores : ",scores)
    print("Mean : ",scores.mean())
    print("Standard deviation : ",scores.std())


housing = load_housing_data()
##  1-Split data
# 1.1 Split it to numerical and categrical attributes
# 1.2 Split it to traning ant test set
####################################################################################
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])

strat_train_set,strat_test_set = makeStratified(housing)
countTest =  strat_test_set["income_cat"].value_counts() / len(strat_test_set)

##### Now I want drop the column of income_cat
for set in (strat_train_set,strat_test_set):
    set.drop(columns=["income_cat"],axis=1,inplace=True)

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housingNum = housing.drop("ocean_proximity",axis=1)

# median =housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median,inplace=True)

#####################################################################################

####   2- Combine feauters and scaling and pipeline
#####################################################################################
########### Custom Transformers    
from sklearn.base import BaseEstimator,TransformerMixin

# rooms_ix,bedrooms_ix,population_ix,households_ix = 3,4,5,6
rooms_ix,bedrooms_ix,population_ix,households_ix = housing.columns.get_loc("total_rooms"),\
    housing.columns.get_loc("total_bedrooms") ,housing.columns.get_loc("population"),housing.columns.get_loc("households")
    
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self,X,y=None):
        return self 
    
    def transform(self,X): ## I cann't send data frame as a parameter here so I send values
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if(self.add_bedrooms_per_room):
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

###########################       Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('attribs_adder',CombinedAttributesAdder()),
    ('stdScaler',StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housingNum) 

from sklearn.compose import ColumnTransformer
num_attribs = list(housingNum)  ### This gives the columns(attributres) of housingNum
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

######################          Select and Train a Model           ###########################
##############    Training and Evaluating on the Training Set
from sklearn.linear_model import LinearRegression
linReg  = LinearRegression()
linReg.fit(housing_prepared,housing_labels)

someData = housing.iloc[0:5]
someLabels = housing_labels.iloc[0:5]
somePreparedData = full_pipeline.transform(someData)
Prediction = linReg.predict(somePreparedData)
# print("Predictions(h(x)) : ",Prediction)
# print("Labels(y) : ",list(someLabels))

######   Measure cost function
from sklearn.metrics import mean_squared_error
housing_predictions = linReg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)  ### Take Y and h(x)
lin_rmse = np.sqrt(lin_mse)
print("Linear regression rmse : ",lin_rmse)

#### This is not good, This is an example of a model underfitting
# the training data. When this happens it can mean that the features do not provide
# enough information to make good predictions, or that the model is not powerful enough.

#### Let’s train a DecisionTreeRegressor. This is a powerful model, capable of finding
# complex nonlinear relationships in the data
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Tree regression rmse : ",tree_rmse) 
####   tree_rmse = 0 thats mean that the model suffer from overfitting
## you don’t want to touch the test set until you are ready to launch a model you are confident about,
##  so you need to use part of the training set for training, and part for model validation.



# ######     Better Evaluation Using Cross-Validation  
# # We have 2 choiced -- 1. Use the train_test_split to split new trainig set
# # -- 2.  Use Scikit-Learn’s K-fold cross-validation feature. it randomly splits the training set into 10 distinct
# # subsets called folds, then it trains and evaluates the Decision Tree model 10 times,
# # picking a different fold for evaluation every time and training on the other 9 folds.
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
# ### Scikit-Learn cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better),
# # so the scoring function is actually the opposite of the MSE (i.e., a negative value), 
# # which is why the preceding code computes -scores before calculating the square root.
# tree_rmse_scores = np.sqrt(-scores) 
# # displayScores(tree_rmse_scores)

# ### We notice now That rmse of linReg is better than rmse pf treeReg so it's suffer from overfitting
# lin_scores = cross_val_score(linReg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
# lin_rmse_score = np.sqrt(-lin_scores)
# # displayScores(lin_rmse_score)

#######   That’s right: the Decision Tree model is overfitting so badly that it performs worse than the Linear Regression model.

###############################  Note : This is a heavy computional code take 5 mins
### let's try another algorithm : the RandomForestRegressor. Random Forests work by training many Decision Trees 
# on random subsets of the features, then averaging out their predictions. Building a model on top of many
# other models is called [ Ensemble Learning ], and it is often a great way to push ML algorithms even further.
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared,housing_labels)
# forest_predictions = forest_reg.predict(housing_prepared)
# forest_score = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv= 10)
# forest_rmse = np.sqrt(-forest_score)
# displayScores(forest_rmse)

############  Random Forests look very promising. However, note that the score on the training set is still 
# much lower than on the validation sets, meaning that the model is still overfitting the training set.


######### I should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both
# the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well.
# This will allow you to easily compare scores across model types,and compare the types of errors they make. You can easily save
# Scikit-Learn models by using Python’s pickle module, or by the joblib library, which is more efficient at serializing large NumPy arrays:
import joblib
# ## after model
# myModel = {'trained parameter': forest_reg,
#                         'cross-validation scores': forest_rmse,
#                         'prdictions': forest_predictions
#                     }

# joblib.dump(myModel,"myModel.pkl")
# # ## then later load model
my_model_loaded = joblib.load("myModel.pkl")  ### now I have all data about forest here