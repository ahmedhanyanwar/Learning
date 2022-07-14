#### chapter 2 until page 62 

import os
import tarfile
from turtle import hideturtle
import urllib.request
from attr import attributes
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from zlib import crc32


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"  ### Must be https://raw.githubusercontent.com/
HOUSING_PATH = os.path.join("datasets","housing") 
HOUSING_URL = DOWNLOAD_ROOT+"datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    os.makedirs(housing_path,exist_ok=True) ## creat a new directory
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)  ### download file from url
    housing_tgz = tarfile.open(tgz_path)  ## open file
    housing_tgz.extractall(housing_path) ## Extract it
    housing_tgz.close() ## close file

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

##   Split set by setting the random number generator’s seed
def split_train_test(data,testRatio):
    np.random.seed(42)  ## To have a same test and train indices
    shuffulIndex = np.random.permutation(len(data))
    testSize = int(len(data)*testRatio)
    testIndex = shuffulIndex[:testSize]
    trainIndex = shuffulIndex[testSize:]
    return data.iloc[trainIndex],data.iloc[testIndex]

### But the seed method fails if we update the data so we will use another way
def testSetCheck(identifier,testRation):
    return crc32(np.int64(identifier))&0xffffffff < testRation* 2**32

def split_train_test_by_id(data,testRatio,idColumn):
    ids = data[idColumn]
    inTestSet = ids.apply(lambda id_:testSetCheck(id_,testRatio))
    return data.loc[~inTestSet],data.loc[inTestSet]

##  make stratified from median_income attribute gives me test set have same numerical calculation of train set
def makeStratified(housing):
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        strat_train_set = housing.loc[train_index] 
        strat_test_set =  housing.loc[test_index]
    return strat_train_set,strat_test_set

# fetch_housing_data()
housing = load_housing_data()
# print(housing.head(4))

# housing.info()

count = housing['ocean_proximity'].value_counts() ### number of each catagory in ocean_proximity column
# count = housing.groupby(['ocean_proximity']).count()
# print(count)

des = housing.describe()  ## gives numberical attributes 
# print(des) 
#####    Draw histogram to see the figures
# housing.hist(bins=50,figsize=(20,15))
# plt.show()


###########   Split test and train set 
####   Methed 1 is to save set and upload it 
####   Methed 2 is to use seed
trainSet , testSet = split_train_test(housing,0.2)

####   Methed 3
## But the seed method fails if we update the data so we will use another way
## Unfortunately, the housing dataset does not have an identifier column. The simplest solution is to use the row index as the ID
housingWithId = housing.reset_index()  ### this will add index colum
trainSet , testSet = split_train_test_by_id(housingWithId,0.2,"index")   ## now identifier is the row index

##    the new data must get appended to the end of the dataset, and no row ever gets deleted. If this is not
##            possible, then you can try to use the most stable features to build a unique identifier.
housingWithId["id"]= housing["longitude"] * 1000 +housing["latitude"]  ## longitude &latitude are a unique feature
trainSet , testSet = split_train_test_by_id(housingWithId,0.2,"id")   

#### Method 4 is to use sklearn-- it's function is look like method 2
from sklearn.model_selection import train_test_split
trainSet , testSet = train_test_split(housing,train_size=0.2,random_state=42)   ##random_state == seed number


#### Method 5 to aviod skewed      Stratified == "طبقية"
##############   make category attribute from median income because I want to divide it into stratum to 
##############   make sure that the attribute have sufficiant information
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])
# housing["income_cat"].hist()
# plt.show()

## the test set generated using stratified sampling has
## income category proportions almost identical to those in the full dataset, whereas the
## test set generated using purely random sampling is quite skewed.
## So using stratified better than using random way
strat_train_set,strat_test_set = makeStratified(housing)
countTest =  strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# print(countTest)

##### Now I want drop the column of income_cat
for set in (strat_train_set,strat_test_set):
    set.drop(columns=["income_cat"],axis=1,inplace=True)
# print(strat_test_set.head(1))  ##  == print(strat_test_set[0:1])

#########################  page 56  ####################3
############### Discover and Visualize the data to gain insight
######### Now we deal with training set only

## create a copy so you can play with it without harming the training set
housing = strat_train_set.copy()
########   Visualize Geographical data  ((latitude and longitude))
housing.plot(kind="scatter",x="longitude",y="latitude")
### Setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points
###  alpha range [0,1]  dafult value is 1 
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
# plt.show()

###   The radius of each circle represents the district’s population (option s)
# ,and the color represents the price (option c).
# We will use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices)
# print(housing.columns)
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100,label="population",figsize=(10,7)\
        ,c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True) ## median_house_value it's a columns contain price
plt.legend()
# plt.show()
### it gives standard correlation coeff (pearson's r) between all attribute
corMat = housing.corr()
## so it gives standard correlation coeff (pearson's r) between price and all other attribute
sort = corMat["median_house_value"].sort_values(ascending=False)
# print(sort)

### to see draw of correlation
from pandas.plotting import scatter_matrix
## I will reduse attribute to be able to see it
attributes =["median_house_value","median_income","total_rooms","housing_median_age"] 
# attributes = list(housing.columns[[-2,-3,3,2]]) ## the same as above line
scatter_matrix(housing[attributes],figsize=(12,8))
# plt.show()

###   The most promising attribute to predict the median house value is the median income,
# so we will zoom in on their correlation scatterplot
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
# plt.show()

############   Combine attributes
# print(housing.columns)
housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]= housing["total_bedrooms"] /housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corMat = housing.corr()
sort =corMat["median_house_value"].sort_values(ascending=False)
print(sort)