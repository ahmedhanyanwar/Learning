from typing import Type
import pandas as pd
# from pandas import read_csv as readpd

#############################################
#'^Pi[a-z]*' contain pi followed by letter from a to z and {*} mean zero or more elements and {^} to be in first not a middle 
#############################################


################   Filtering data (LOC) #################
#########################################################
##### in filter pandas we use    & | ~ instead of and or not
#####    and or not will give error
df = pd.read_csv('Modified.csv') #data frame ###pokemon_data.csv

filter = df.loc[(df['Type 1']=='Grass') & (df['Type 2'] =='Poison')&(df['HP']>70)]

###  the new filter data save old index in it to reset index we will use

# filter = filter.reset_index()   ### this will make new index and save old index
# filter = filter.drop(columns='index')
filter = filter.reset_index(drop=True,inplace=True)  ##  ==  filter = filter.reset_index() + filter.drop(colums='index')

### If I want name which cotain Mega -- turn it to string then use contain
conMega = df.loc[df['Name'].str.contains('Mega')]

### If I want check if Type1 == Grass or Fire   -- I will use re library
## Reglar Expression  it is super powerful in filtering data based on certain textual patterns
import re   
fireGrass = df.loc[df['Type 1'].str.contains('Fire|Grass',regex=True)]
#### It is fire and grass with small letter but it works because of re.I
fireGrass = df.loc[df['Type 1'].str.contains('fire|grass',flags=re.I,regex=True)]  

####  contain pi followed by letter from a to z and {*} mean zero or more elements and {^} to be in first not a middle 
nameWithPi = df.loc[df['Name'].str.contains('^Pi[a-z]*',flags=re.I,regex=True)]
#########################################################



################   Conditional changes  #################
#########################################################
### df.loc[row==fire,colum of Type 1] = 'Flamer'
df.loc[df['Type 1']=='Fire','Type 1'] = 'Flamer'
df.loc[df['Type 1']=='Flamer','Type 1'] = 'Fire'

###   convert legendary of rows thats Type 1 == Fire to True
df.loc[df['Type 1']=='Fire','Legendary'] = True

###  Multiple changes
df.loc[df['Total'] > 500,['Generation','Legendary']] = 'Test value'
df.loc[df['Total'] > 500,['Generation','Legendary']] = ['Test1','Test2']

#########################################################


#######    Aggregate Statistics (Groupby)   #############
#########################################################
df = pd.read_csv('Modified.csv')

## I want find the type 1 average to see who has high Defense
## it grouped all different types in type 1
highDef = df.groupby('Type 1').mean().sort_values('Defense',ascending=False)
highAtt  = df.groupby('Type 1').mean().sort_values('Attack',ascending=False)

## I want to know count [number of pokemon in type 1]
num = df.groupby(['Type 1']).count() # but there are som NAN in some colums so I will do:
num = df['Type 1'].value_counts()  # but number of each member in Type 1 regardless colums
# print(num)
##  To concern in count rather than blanks
df['Count'] = 1
num  = df.groupby(['Type 1']).count()['Count']
df = df.drop(columns='Count') ## I will remove count colums because I don't need it now
# print(num)
#########################################################


#######    Working with large amounts of data   #############
#############################################################
new_df = pd.DataFrame(columns=df.columns) ### empty data frame has same colums as df
for df in pd.read_csv('Modified.csv',chunksize=5):
    results = df.groupby(['Type 1']).count()
    new_df = pd.concat([new_df,results])

    # print("CHUNKS = ")
    # print(df)  ### while print 5 by 5 rows     
# print(new_df)  #### now we rdused data by another useful info

#############################################################
