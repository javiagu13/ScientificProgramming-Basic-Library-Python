--------------------------------------------------------------
Installation (run command in python prompt):

```
pip install SciProgPackage==1.2.0
```

Python package (in the original PyPI website with more instructions):

```
https://pypi.org/project/SciProgPackage/1.2.0/
```

---------------------------------------------------------------

Testing and Usage instructions:

In order to import the library simply on the top of the file import it as follows:

```
from SciProgPackage.ScientificProgramming import SciProg as sp
```

In order to run the different functions:

```
sp.NAMEOFTHEFUNCTION(attributes)
```

As an example:

```
dat = np.arange(1,11)
discrete_dat, cutoff = sp.atributeDiscretizeEF(dat, 3)
```
---------------------------------------------------------------------------
In order to have further explanation, download the following file from the repository:

```
pyVignette.html
```
---------------------------------------------------------------------------------

Finally in order to run all the possible tests here is a code to test it:

It is the same code as contained in main.py, so in case you do not want to copy the content
simply download main.py file and run on the command prompt:

```
python main.py
```

```
# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import collections
import math
import scipy 
from numpy import genfromtxt
import pandas as pd

from SciProgPackage.ScientificProgramming import SciProg as sp



##TESTS
#TEST
print("TEST1")
print(" ")
print("atributeDiscretizeEF")
print("data:")
dat = np.arange(1,11)
print(dat)
print("RESULT:-------------------")
discrete_dat, cutoff = sp.atributeDiscretizeEF(dat, 3)
print("discrete_dat: ", discrete_dat)
print("cutoff: ", cutoff)
print("--------------------------")
print(" ")


print("TEST2")
print(" ")
#TEST
print("datasetDiscretizeEF")
print("data:")
data=np.random.randint(10,size=(10,10))
print(data)
print("RESULT:-------------------")
print(sp.datasetDiscretizeEF(data,5))
print("--------------------------")
print(" ")

print("TEST3")
print(" ")
#TEST
print("atributeDiscretizeEW")
print("data:")
dat = np.arange(1,11)
print(dat)
discrete_dat, cutoff = sp.atributeDiscretizeEW(dat, 3)
print("RESULT:-------------------")
print("discrete_dat: ", discrete_dat)
print("cutoff: ", cutoff)
print("--------------------------")
print(" ")

print("TEST4")
print(" ")
#TEST
data=np.random.rand(10,10)
print("datasetDiscretizeEW")
print("dat: ",data)
print("RESULT:-------------------")
print(sp.datasetDiscretizeEW(data,5))
print("--------------------------")
print(" ")

print("TEST5") 
print(" ")
print("variance")
#TEST
print("data")
numberCol=np.random.rand(10)
print(numberCol)
print("RESULT:-------------------")
print(sp.variance(numberCol))
print("--------------------------")
print(" ")
 
print("TEST6")
print(" ")
print("auc")
print("data")
#TEST
numberCol=np.random.rand(10)
numberCol
boolCol=np.random.randint(0,2,size=10)
boolCol
print(numberCol)
print(boolCol)
result=sp.auc(numberCol,boolCol)
print("RESULT:-------------------")
print(result)
print("--------------------------")
print(" ")
 
print("TEST7")
print(" ")
print("datasetEntropy")
 #TEST
numberCol=np.random.rand(10)
boolCol=np.random.randint(0,2,size=10)
data=np.column_stack((numberCol,boolCol))
print("data")
print(data)
print("RESULT:-------------------")
val=sp.datasetEntropy(data)
print(val) 
print("--------------------------")
print(" ")

print("TEST8")
print(" ")
print("variableNormalization")
print("data:")
print(np.array([1,2,3,4,5,5,65,4,3]))
 #TEST
print("RESULT:-------------------")
data=sp.variableNormalization(np.array([1,2,3,4,5,5,65,4,3]))
print(data) 
print("--------------------------")
print(" ")


print("TEST9")
print(" ")
print("variableEstandarization")
print("data:")
print(np.array([1,2,3,4,5,5,65,4,3]))
 #TEST
print("RESULT:-------------------")
data=sp.variableEstandarization(np.array([1,2,3,4,5,5,65,4,3]))
print(data)
print("--------------------------")
print(" ")

print("TEST10")
print(" ")
print("datasetNormalization")
#TEST
data=np.random.rand(10,10)
a=np.array([1,2,3,4,5,5,65,4,3])
b=np.array([3,2,6,4,99,5,25,42,1])
data=np.column_stack((a,b))
print("data:")
print(data)
print("RESULT:-------------------")
norm=sp.datasetNormalization(data.astype(float))
print(norm) 
print("--------------------------")
print(" ")

print("TEST11")
print(" ")
print("datasetEstandarization")
 #TEST
data=np.random.rand(10,10)
a=np.array([1,2,3,4,5,5,65,4,3])
b=np.array([3,2,6,4,99,5,25,42,1])
data=np.column_stack((a,b))
print("data:")
print(data)
print("RESULT:-------------------")
norm=sp.datasetEstandarization(data.astype(float))
print(norm) 
print("--------------------------")
print(" ")

print("TEST12")
print(" ")
print("filterDataset")
 #TEST
data=np.random.rand(10,10)
a=np.array([1,2,3,4,5,5,65,4,3])
b=np.array([1,2,3,4,5,56,65,4,3])
c=np.array([3,2,6,4,99,5,25,42,1])
data=np.column_stack((a,b,c))
print("data:")
print(data)
print("RESULT:-------------------")
val=sp.filterDataset(np.array(data.astype(float)),10000,"variance")
print(val)
print("--------------------------")
print(" ")

print("TEST13")
print(" ")
print("atributesCorrelation")
 #TEST
data=np.random.rand(10,10)
a=np.array([1,2,3,4,5,5,65,4,3])
b=np.array([3,2,6,4,99,5,25,42,1])
b=np.array([3,4,4,4,9,5,25,42,1])
data=np.column_stack((a,b,c))
print("data:")
print(data)
print("RESULT:-------------------")
norm=sp.atributesCorrelation(data.astype(float))
print(norm) 
print("--------------------------")
print(" ")

print("TEST14")
print(" ")
print("plotAUC")
 #TEST
numberCol=np.random.rand(10)
boolCol=np.random.randint(0,2,size=10)
print("data:")
print(numberCol)
print("data:")
print(boolCol)
print("RESULT:-------------------")
result=sp.plotAUC(numberCol,boolCol)
print(result)
print("--------------------------")
print(" ")

print("TEST15") 
print(" ")
print("plotMutualInformation")
data=np.random.rand(10,2)
print("data:")
print(data)
print("RESULT:-------------------")
print(sp.plotMutualInformation(data))
print("--------------------------")
print(" ")

print("TEST16")
print(" ")
print("datasetRead")
print("RESULT:-------------------")
data=sp.datasetRead('/content/myData.csv') 
print(data)
print("--------------------------")


print("TEST17")
print(" ")
print("writeDatasetCSV")
data=np.random.rand(10,2)
print("data:")
print(data)
print("RESULT:-------------------")
print(sp.writeDatasetCSV(data,'newData.csv'))
print("--------------------------")

```


