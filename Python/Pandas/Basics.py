import pandas as pd

###   Reading file  & Sorting/Describing Data  & Making changes to the data & Saving our Data 
#### Video link : https://www.youtube.com/watch?v=vmEHCJofslg

#####################################   Reading file
df = pd.read_csv('pokemon_data.csv')
df_txt = pd.read_csv('pokemon_data.txt',delimiter='\t') ## to separate data
df_xlsx = pd.read_excel('pokemon_data.xlsx')

firstPart =  df.head(3) ## subset of df 
lastPart =  df.tail(3) ## subset of df 
# print(df_txt.head(2))

### Read headers
col = df.columns  ## like  name type ... etc

### Read each columns
col = df['Name'][0:5] ## [0:5] read part of them
col = df.Name[0:5]    ## [0:5] read part of them
cols = df[['Name','Speed']][0:5]  ## read multiple column
# print(cols)

### Read each Row
toalRow = df.iloc   # iloc == Index LOCation
row = df.iloc[1]    # second Row
# print(toalRow[1:4])

### Read a specific position (Row,col)
item =df.iloc[2,1]
# print(item)

###  Reading it as a numpy array
values = df.values
# print(values)

###  To find specific row depends on multiple condition
###  we can say we search about something 
fireRow = df.loc[df['Type 1'] == 'Fire']
# print(fireRow)

### acess data row by row
# for index ,row in df.iterrows():
    # print(index,row['Name'])   ## Print name in each row



##########      Sorting/Describing Data    ########
###################################################
##    df.describe()   Gives us higher calcuation like mean & std & count & min & max
des = df.describe()

##Sort
sort_aToz = df.sort_values('Name') ##  Sort it alphabetically from A to Z
sort_zToa = df.sort_values('Name',ascending=False) ##  Sort it alphabetically from Z to A

## Sort by multiple things == sort by first then second
sort_mult = df.sort_values(['Type 1','HP'])
sort_multde = df.sort_values(['Type 1','HP'],ascending=[True,False]) ##  Sort Type1 alphabetically from A to Z and HP from High to Low



###################################################


##########      Making changes to the data    ########
######################################################
## add another column called tatal and assign it's valuse
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
df = df.drop(columns=['Total'])  ## To delete columns 

## Another way to add colums with easiest way
HPIndex  =  df.columns.get_loc('HP')  ## git index of column
speedIndex = df.columns.get_loc('Speed')
df['Total'] = df.iloc[:,HPIndex:speedIndex+1].sum(axis=1)  #axis = 1 because we want sum on X axis

### Reorder colums total   before  HP
cols = list(df.columns)
df = df[cols[0:HPIndex] + [cols[-1]]+cols[HPIndex:12]]

print(df.head(5))
######################################################

###############      Saving our Data    ##############
######################################################
# df.to_csv('Modified.csv',index=False) ## index = false to remove index from new file
# df.to_excel('Modified.xlsx',index=False)
# df.to_csv('Modified.txt',index=False,sep='\t')  ## sep ='\t' to seprate output

######################################################

############## calcualte Standard correlation coefficient 
corrMat = df.corr()
# print(corrMat)
