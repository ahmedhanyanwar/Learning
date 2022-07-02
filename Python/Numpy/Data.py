import numpy as np
####### data
data = np.genfromtxt('data.txt',dtype='int32',delimiter=',')
data = data.astype('int16') ## changing type
# print(data)

############################## Boolean Masking and Advanced Indexing
res = data[data>50] ## Element that has a value > 50
res = np.any(data > 50,axis = 0)  ## if there is an element > 50 in each COLUMS
res = np.all(data > 50,axis = 0)  ## if all elements in each COLUMS > 50
res = data[((data>50) & (data<100))]
# print("REs =",res)

#### I can index using list 
arr = np.array([1,2,3,4,5,6,7,8,9])
res = arr[[1,2,5]]
# print("REs =",res)

arr = np.zeros((6,5),dtype='int32')
x= 1
for i in range(0,6):
    for j in range(0,5):
        arr[i,j] = x
        x+=1
    
print(arr[[0,1,2,3],[1,2,3,4]])
print(arr[2:4,0:2])
print(arr[[0,-2,-1],3:])

