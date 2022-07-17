#### chapter 2 from 63 to 71
import os
from joblib import PrintTime
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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

# def fillMissing(housing):
#     #### 3.2 using scikit-learn 
#     from sklearn.impute import SimpleImputer
#     # determine the strategy you want fill by it
#     imputer = SimpleImputer(strategy="median")
#     # we will delete ocean_proximity attribute because median is computed for numerical
#     housingNum = housing.drop("ocean_proximity",axis=1)
#     # now compute madian for all attribute
#     imputer.fit(housingNum)
#     # To see the median values
#     median = imputer.statistics_

#     # Use this “trained” imputer to transform the training set by replacing missing values by the learned medians
#     # X is a plain Numpy array containing the transformed features
#     X = imputer.transform(housingNum)
#     # put it back into a Pandas DataFrame
#     ## this will create new data frame contain colmns of housingNum and filled with X
#     housing_tr = pd.DataFrame(X,columns=housingNum.columns) 
#     #####   Now housing_tr doesnot cotain any Null data
#     return housing_tr

# fetch_housing_data()
housing = load_housing_data()

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])

strat_train_set,strat_test_set = makeStratified(housing)
countTest =  strat_test_set["income_cat"].value_counts() / len(strat_test_set)

##### Now I want drop the column of income_cat
for set in (strat_train_set,strat_test_set):
    set.drop(columns=["income_cat"],axis=1,inplace=True)



#####################   Prepare the Data for Machine Learning Algorithms #####################3
########## separate the predictors and the labels since we don’t necessarily want to apply
#### the same transformations to the predictors and the target values
# (note that drop() creates a copy of the data and does not affect strat_train_set)
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

########### Data Cleaning
###Most Machine Learning algorithms cannot work with missing features
##  housing.info()
# We have three options:
# 1.Get rid of the corresponding districts. 
housing.dropna(subset=["total_bedrooms"])  ### dropna  delete whole row which contain missing value
#  2.Get rid of the whole attribute.
housing.drop("total_bedrooms",axis=1)  
#   3.Set the values to some value (zero, the mean, the median, etc.).
#### 3.1 using pandas
median =housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median,inplace=True)

#### 3.2 using scikit-learn 
from sklearn.impute import SimpleImputer
# determine the strategy you want fill by it
imputer = SimpleImputer(strategy="median")
# we will delete ocean_proximity attribute because median is computed for numerical
housingNum = housing.drop("ocean_proximity",axis=1)
# now compute madian for all attribute
imputer.fit(housingNum)
# To see the median values
median2 = imputer.statistics_
# if we calculate median by pandas
median3 = housingNum.median().values ## Same as median2
# print(median2-median3)

# Use this “trained” imputer to transform the training set by replacing missing values by the learned medians
# X is a plain Numpy array containing the transformed features
X = imputer.transform(housingNum)
# put it back into a Pandas DataFrame
## this will create new data frame contain colmns of housingNum and filled with X
housing_tr = pd.DataFrame(X,columns=housingNum.columns) 
#####   Now housing_tr doesnot cotain any Null data

####  Rathar than using fit() then transform I can use fit_transform() 
# [sometimesfit_transform() is optimized and runs much faster)]


####################    Handling Text and Categorical Attributes   #########################
## Better to turn it into numerical Attributes
housingCategory = housing[["ocean_proximity"]]
print(housingCategory.head(10))

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
ordinal_encoder = OrdinalEncoder()
### turn to number
housingCategoryEncoded = ordinal_encoder.fit_transform(housingCategory)
# print(housingCategoryEncoded[0:10])

########  To know catogries we know it by public instance variable_
cat = ordinal_encoder.categories_
# print(cat)


#### One issue with this representation is that ML algorithms will assume that two nearby
## values are more similar than two distant values
# To fix this issue, a common solution is to create one binary attribute per category: one attribute
# equal to 1 when the category is “<1H OCEAN” (and 0 otherwise)  This is
# called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold).
catEncoder = OneHotEncoder()
housingCategory1Hot = catEncoder.fit_transform(housingCategory)
# print(type(housingCategory1Hot)) ###<class 'scipy.sparse.csr.csr_matrix'> to save memory
# print(housingCategory1Hot) 

#### To turn it into dense(Numpy) array we use toarray() method
housingCat1HotArray = housingCategory1Hot.toarray()
# print(housingCat1HotArray) 
cat = catEncoder.categories_
# print(cat)


################################ Custom Transformers    ############################
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

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housingExtraAttribs = attr_adder.transform(housing.values)
col = list(housing.columns)
col.append('rooms_per_household')
col.append('population_per_household')
housingExtraAttribs = pd.DataFrame(housingExtraAttribs,columns=col)
# print(housingExtraAttribs.head(10)) 
##############################################################################3

# des = housing.describe()
# # print(des.head()[3:4])  ## Min
# # print(des.tail()[4:5]) ## Max

###########################       Feature scaling
####### 1- min-max scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
scale_minMax = MinMaxScaler()
X = scale_minMax.fit_transform(housingNum)
housing_scale_minMax = pd.DataFrame(X,columns=housingNum.columns)
# print(housing_scale_minMax.head(10))

####### 2- Standardization
from sklearn.preprocessing import StandardScaler
scal_std = StandardScaler()
X = scal_std.fit_transform(housingNum)
housing_scaleStd = pd.DataFrame(X,columns=housingNum.columns)
# print(housing_scaleStd.head(10))


###########################       Transformation Pipelines
## The Pipeline constructor takes a list of name/estimator pairs defining a sequence of
# steps. All but the last estimator must be transformers (i.e., they must have a fit_transform() method).
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('attribs_adder',CombinedAttributesAdder(add_bedrooms_per_room=False)),
    ('stdScaler',StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housingNum) 

##  We must handle the categorical attribs and numerical in different way
##  To solve this problem we will use this
from sklearn.compose import ColumnTransformer

num_attribs = list(housingNum)  ### This gives the columns(attributres) of housingNum
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs)
])
# col = list(housing)
# ### This cat because of oneHotEncoder
# col.append('rooms_per_household')
# col.append('population_per_household')
# col.append('Cat0')  
# col.append('Cat1')
# col.append('Cat2')
# col.append('Cat3')
print(housing.head(10))
housing_prepared = full_pipeline.fit_transform(housing)
# housingDF = pd.DataFrame(housing_prepared,columns=col)
# print(housingDF.head(10))s
