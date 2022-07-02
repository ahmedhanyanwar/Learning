import numpy as np 

#sin
PI = 3.141592653589793

a = np.array([90,90,90])*PI / 180
sin_ = np.sin(a)

##################################   Linear algebra
# matrix multiplications
a = np.ones((3,2))
b = np.full((2,3),2)
res = np.matmul(a,b)
res = np.transpose(a)

# determinant & inverse  of matrix   linalg = LINear ALGebra
I = np.random.randint(0,10,size=(3,3))
res =  np.linalg.det(I)
res = np.linalg.inv(I)
res = np.linalg.eigvals(I)
##################################

##################################   statistics
arr = np.array([[1,2,3],[4,5,6]])

###   Minimum
res = np.min(arr)  ## min in whole array
resCOL = np.min(arr,axis=0)  ## axis= 0 find min in each columns
resROW = np.min(arr,axis=1)  ## axis= 1 find min in each rows

###   Maximum
res = np.max(arr) #6
resCOL = np.max(arr,axis=0) #4,5,6  
resROW = np.max(arr,axis=1)  #3,6
# print(res,resCOL,resROW)

###    SUM
res = np.sum(arr) #21
resCOL = np.sum(arr,axis=0) #5,7,8  
resROW = np.sum(arr,axis=1)  #6,15
# print(res,resCOL,resROW)
##################################


##################################   Reorganization
##  Reshaping  size is so imp
arr = np.array([[1,2,3,4],[5,6,7,8]])
res = arr.reshape((8,1))

## Stacking or concatenation  size is so imp
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
res = np.vstack([v1,v2])  ######  MUST put in   [  ]
res = np.hstack([v1,v2])  ######  MUST put in   [  ]

print(res)
##################################