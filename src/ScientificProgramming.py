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

class ScientificProgramming:
    #pass

    
    ######################################Discretization##################################
    def atributeDiscretizeEF(atribute, n_bins):
        a = len(atribute)
        n = int(a / n_bins)
        kont=0
        arr = []
        for i in range(0, n_bins):
            kont+=1
            for j in range(i * n, (i + 1) * n):
                if j >= a:
                    break
                arr = arr + [kont]
        return(arr)
        
    """ #TEST
    dat = np.arange(1,11)
    discrete_dat = atributeDiscretizeEF(dat, 3)
    print("dat: ", dat)
    print("discrete_dat: ", discrete_dat)
     """


    def datasetDiscretizeEF(data, n_bins):
        for i in range(0,len(data[1,])):
            discrete = ScientificProgramming.atributeDiscretizeEF(data[:,i],n_bins)
            print(discrete)
            col=discrete.copy()
            data[:,i]=col
        return data
    """ #TEST
    data=np.random.rand(10,10)
    datasetDiscretizeEF(data,5)
    print("dat: ",data) """


    def atributeDiscretizeEW(atribute, n_bins):
      split = np.array_split(np.sort(atribute), n_bins)
      cutoffs = [x[-1] for x in split]
      cutoffs = cutoffs[:-1]
      discrete = np.digitize(atribute, cutoffs, right=True)
      return discrete, cutoffs

    """ #TEST
    dat = np.arange(1,11)
    discrete_dat, cutoff = atributeDiscretizeEW(dat, 3)
    print("dat: ", dat)
    print("discrete_dat: ", discrete_dat)
    print("cutoff: ", cutoff)
     """




    def datasetDiscretizeEW(data, n_bins):
      for i in range(0,len(data[1,])):
        discrete, cutoff = ScientificProgramming.atributeDiscretizeEW(data[:,i],n_bins)
        data[:,i]=discrete
      return data


    """ #TEST
    data=np.random.rand(10,10)
    datasetDiscretizeEW(data,5)
    print("dat: ",data)
     """

    ######################################Metric Calculation##################################
    def datasetVariance(data):
      arr=[]
      for i in range(0,len(data[1,])):
        val = np.var(data[:,i])
        arr.append(val)
      return arr
    """ 
    #TEST
    data=np.random.rand(10,10)
    varList=datasetVariance(data)
    print("variances: ",varList)
     """



    def atributesAUC(data, threshold):
      for i in range(0,len(data[0,:])):
        if(data[i,0]<=threshold):
          data[i,0]=0
        else:
          data[i,0]=1
      data=data.astype(int)
      return(roc_auc_score(data[:,0], data[:,1]))

    """ #TEST
    numberCol=np.random.rand(10)
    numberCol
    boolCol=np.random.randint(0,2,size=10)
    boolCol
    data=np.column_stack((numberCol,boolCol))
    data

    result=atributesAUC(data,0.4)
    print(result)
     """


    def datasetEntropy(data):
      arr=[]
      for i in range(0,len(data[1,])):
        val = ScientificProgramming.entropy(data[:,i])
        arr.append(val)
      return arr


    def entropy(arr):
      auxArr=[]
      frequencyArr=[]
      freq = collections.Counter(arr)
      for (key, value) in freq.items():
          auxArr.append(value)
      #get the sum of the array
      suma=0
      for i in auxArr:
        suma+=i
      #get the frequency of each element
      for i in range(0,len(auxArr)):
        auxArr[i]=auxArr[i]/suma
      auxArr
      #calculate the entropy
      total=0
      for i in auxArr:
        total=total-i*math.log(i, 2.0)
      return total

    """ #TEST
    numberCol=np.random.rand(10)
    numberCol
    boolCol=np.random.randint(0,2,size=10)
    boolCol
    data=np.column_stack((numberCol,boolCol))
    data

    val=datasetEntropy(data)
    print(val) """


    ######################################Normalization and Estandarization##################################
    def variableNormalization(v):
      vnorm = (v - np.amin(v)) / (np.amax(v) - np.amin(v))
      return(vnorm)

    """ #TEST
    data=variableNormalization(np.array([1,2,3,4,5,5,65,4,3]))
    print(data) """

    def variableEstandarization(v):
      vest = (v-np.mean(v)) / np.std(v)
      return(vest)
      
    """ #TEST
    data=variableEstandarization(np.array([1,2,3,4,5,5,65,4,3]))
    print(data)
     """


    def datasetNormalization(data):
      arr=[]
      for i in range(0,len(data[1,])):
        data[:,i] = ScientificProgramming.variableNormalization(data[:,i])
        print(data)
      return data

    """ 
    #TEST
    data=np.random.rand(10,10)
    a=np.array([1,2,3,4,5,5,65,4,3])
    b=np.array([3,2,6,4,99,5,25,42,1])
    data=np.column_stack((a,b))
    norm=datasetNormalization(data.astype(float))
    print(norm) """



    def datasetEstandarization(data):
      arr=[]
      for i in range(0,len(data[1,])):
        data[:,i] = ScientificProgramming.variableEstandarization(data[:,i])
        print(data)
      return data

      
    """ #TEST
    data=np.random.rand(10,10)
    a=np.array([1,2,3,4,5,5,65,4,3])
    b=np.array([3,2,6,4,99,5,25,42,1])
    data=np.column_stack((a,b))
    norm=datasetEstandarization(data.astype(float))
    print(norm) """


    ######################################Filtering based on Metrics##################################
    def filterVariance(data, var):
      
      #getting vector of variancess
      vec=ScientificProgramming.datasetVariance(data)
      columnsToDelete=[]
      for i in range(0,len(vec)):
        if(var>=vec[i]):
          columnsToDelete.append(1)
        else:
          columnsToDelete.append(0)

      print(columnsToDelete)
      #deleting columns
      for i in range(len(vec)-1,-1,-1):
        print(i)
        if(np.var(columnsToDelete) == 0 and columnsToDelete[i]==1):
          data=[]
        elif(columnsToDelete[i]==1):
          data = np.delete(data, 0, i)
          #data = np.delete(a, 0, i)

      return(data)


    """ #TEST
    data=np.random.rand(10,10)
    a=np.array([1,2,3,4,5,5,65,4,3])
    b=np.array([3,2,6,4,99,5,25,42,1])
    data=np.column_stack((a,b))
    val=filterVariance(np.array(data.astype(float)),10000)
    print(val)


    data=np.random.rand(10,10)
    a=np.array([1,2,3,4,5,5,65,4,3])
    b=np.array([3,2,6,4,99,5,25,42,1])
    data=np.column_stack((a,b))
    print(np.var(data[:,0]))
    print(np.var(data[:,1])) """




    ######################################Correlation calculus by pairs##################################
    def atributesCorrelation(data):
      return scipy.stats.spearmanr(data[:,0], data[:,1])

    """ #TEST
    data=np.random.rand(10,10)
    a=np.array([1,2,3,4,5,5,65,4,3])
    b=np.array([3,2,6,4,99,5,25,42,1])
    data=np.column_stack((a,b))
    norm=atributesCorrelation(data.astype(float))
    print(norm) """



    ######################################Plots for AUC and Mutual Information##################################
    def plotAUC(data,threshold):
      for i in range(0,len(data[0,:])):
        if(data[i,0]<=threshold):
          data[i,0]=0
        else:
          data[i,0]=1
      data=data.astype(int)
      lr_fpr, lr_tpr, _ = roc_curve(data[:,0], data[:,1])
      # plot the roc curve for the model
      plt.plot(lr_fpr, lr_tpr, marker='.', label='Curve')
      # axis labels
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      # show the legend
      plt.legend()
      # show the plot
      plt.show()

    """ #TEST
    numberCol=np.random.rand(10)
    numberCol
    boolCol=np.random.randint(0,2,size=10)
    boolCol
    data=np.column_stack((numberCol,boolCol))
    data

    result=plotAUC(data,0.4)
    print(result)
     """




    def plotMutualInformation(data):
      plt.title("mutual information")
      plt.xlabel('X - value')
      plt.ylabel('Y - value')
      plt.scatter(data[:,0], data[:,1])

    """ data=np.random.rand(10,2)
    plotMutualInformation(data) """



    ######################################Write and read datasets##################################

    #TEST
    def datasetRead(root):
      my_data = genfromtxt(root, delimiter=',')
      return my_data

    """ print(datasetRead('/content/myData.csv'))
    data=datasetRead('/content/myData.csv') """

    def writeDatasetCSV(data, root):
      data=pd.DataFrame(data)
      data.to_csv('/content/myData2.csv', sep=',')

    #Test
    # writeDatasetCSV(data, '/content/myData2.csv')