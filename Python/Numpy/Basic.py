## link of video : https://www.youtube.com/watch?v=GB9ByFAIAH4
import numpy as np
###############################           Basics       #################################
########################################################################################
# declearation
a = np.array([1,2,3,4,5,6],dtype='int32') ## as default it is int32

b = np.array([[1.1,2.1,3.5],[4.1,5.2,6.0]],dtype='float64') ## as default it is float64 

#get dimension Must br a squre matrix
print("Dimension of a = ",a.ndim)
print("Dimension of b = ",b.ndim)

#get shape
print("Shape of a = ",a.shape)
print("Shape of b = ",b.shape)

#get type
print("Type of a = ",a.dtype)
print("Type of b = ",b.dtype)

#get size( size of element in bytes)
print("Size of single element in a = ",a.itemsize," Bytes")
print("Size of single element in b = ",b.itemsize," Bytes")

#get total size in bytes  =  a.itemsize * a.size
print("Total size of a = ",a.nbytes," Bytes")
print("Total size of b = ",b.nbytes," Bytes")

#get number of total element
print("Number of elements of a = ",a.size)
print("Number of elements of b = ",b.size)
########################################################################################


###################  Accessing/Changing specific elements  #############################
########################################################################################
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])

print(a[0,:]) # first row
print(b[0,:,:]) # first 2nd dimension array

print(a[:,1]) # second column
print(b[:,1,:]) # second row in each 2nd dimension array

# a[row, startIndex:Endindex:step]
print(a[0, 1:3:2])

print(a)
a[:,1] = [10,10]
print(a)

print("B array before : ")
print(b)
b[0,:,1:3] = [[10,10],[10,10]]
print("B array after : ")
print(b)
########################################################################################

###################  Accessing/Changing specific elements  #############################
########################################################################################
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])

print(a[0,:]) # first row
print(b[0,:,:]) # first 2nd dimension array

print(a[:,1]) # second column
print(b[:,1,:]) # second row in each 2nd dimension array

# a[row, startIndex:Endindex:step]
print(a[0, 1:3:2])

print(a)
a[:,1] = [10,10]
print(a)

print("B array before : ")
print(b)
b[0,:,1:3] = [[10,10],[10,10]]
print("B array after : ")
print(b)
########################################################################################

###################  Initializing Different Types of Arrays  ###########################
########################################################################################

#All 0s matrix
a = np.zeros((4,4))

#All 1s matrix
b = np.ones((5,5,5),dtype='int32')

#All other number    np.full((shape),number)
c= np.full((3,3),5)

#All other number (full_like) array has a same size of anthor array
d = np.full(a.shape,5)  #or
d = np.full_like(a,5)

# Random decimal number
e = np.random.rand(4,4)

# or 
e = np.random.random_sample(a.shape)

# Random int number    np.random.randint( low ,high ,size = (shzpe)   )  results do not include High value
e = np.random.randint(0,2,size=(3,3))

# Identity matrix
I = np.identity(5)

# Repeat matrix np.repeat( array , number of repeation , axis = X or Y)
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3,axis=0)
# print(r1)

########################################################################################

###########  np.c_
##Translates slice objects to concatenation along the second axis.
arr = np.c_[np.array([1,2,3]),np.array([4,5,6])]
print("array np.c_ : ")
# print(arr)