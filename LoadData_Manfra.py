# -*- coding: utf-8 -*-
"""
Created on 7/23/2019

@author: Tailung Wu

verstion: 3.1

Package requirement:
1. Plase make sure QCoDes enviroment is enabled
2. Pandas, NumPy package are installed

"""

########################
##Package
########################
import pandas as pd
import scipy.constants as sc
import numpy as np
from qcodes.dataset.data_set import load_by_id

#########################

##########################################
## Load data from QCodes Dataset        ##
##########################################

def Load_data_QCodesDB(Run_id, # exp. run ID in database
                       Cols=[]):  # rename column names
    """
    Load single experiment run from QCodes Dataset into a pandas data frame\n
    ===========\n
    Parameters:\n
    Run_id : experiment run ID in dataset. Run ID can be checked by "experiments()" command \n
    Cols : array of column names.\n
    \n
    Return Pandas dataframe \n
    """
    tmp=load_by_id(Run_id).get_data_as_pandas_dataframe()
    i=0
    for s in tmp.values():
        if i==0:
            df=s
        else:
            df[s.columns[0]]=s.iloc[:,0]
        i+=1
        
    df=df.reset_index()
    if Cols != []:
        df.columns=Cols
    
    del i,s,tmp
    return df



##########################################
## Load data from QCodes program        ##
##########################################
def Load_QCodes(filename,Cols=[]):
    """
    Load data from QCodes program into a pandas data frame\n
    Parameters:\n
    ===========\n
    Filename: Qcodes data file\n
    Cols : array of column names, Example ['V','I','R']\n
    \n
    Return Pandas dataframe \n
    """
    df=pd.read_csv(filename,sep='\t',skiprows=[0,2],header=0)
    #df.rename(columns={df.columns[0]: df.columns[0][3:-1]}, inplace=True)
    if Cols != []:
        df.columns=Cols

    return df

#################################################
## Load Jimmy's pajama data  into X,Y,Z column ##
#################################################
def Load_XYZ(filename,Cols):
    """
    Load Jimmy's pajama data into one matrix in pandas data frame\n
    Parameters:\n
    ===========\n
    Filename: pajama data file\n
    Col_labels : array of column names, Example ['V','I','R']\n
    1st Column: major variable (x) ; 2nd Column : secondary variable(y); 3rd Column : Data\n
    \n
    Return Pandas dataframe \n
    """
    df=pd.read_csv(filename,sep='\n',names=['A'])
    foo = lambda x: pd.Series([i for i in x.split('\t')[:-1]])
    data = df['A'].apply(foo) # row line slice
        
    boo = lambda x: pd.Series([i for i in x.split('|')[:len(Cols)]])
    # Extract index list
    Out = pd.DataFrame(columns=Cols)
    
    for i in range(len(data.index)):
        rev2 = data.iloc[i,:].dropna(axis=0,how='any').apply(boo)
        rev2.columns=Cols
        Out=Out.append(rev2,ignore_index=True)
    
    for i in range(len(rev2.columns)):
        Out.iloc[:,i]=pd.to_numeric(Out.iloc[:,i])    
    
    return Out


##########################################
## Load Jimmy's pajama data into Matrix ##
##########################################
def Load_Matrix(filename,X_label='',Y_label='',dataName='',z=2,x=0,y=1):
    """
    Load Jimmy's pajama data into one matrix in pandas data frame\n
    Parameters:\n
    ===========\n
    Filename: pajama data file\n
    z : Column index for z (default : 2)\n
    x : Column index for x (default : 0)\n
    y : Column index for y (default : 1)\n\n
    X_label : name of X\n
    Y_label : name of y\n
    dataName : name of dataframe\n
    \n
    Return Pandas dataframe \n
    """
    df=pd.read_csv(filename,sep='\n',names=['A'])
    foo = lambda x: pd.Series([i for i in reversed(x.split('\t'))])
    data = df['A'].apply(foo)
    data.drop(columns=[0],inplace=True)
    
    boo = lambda x: pd.Series([i for i in (x.split('|'))])
    # Extract index list
    rev2 = data.iloc[:,0].apply(boo)
    idx=pd.to_numeric(rev2.iloc[:,y])  #Extract index list ; y
    
    cols=[]
    for i in range(len(data.columns)):
        rev2 = data.iloc[:,i].apply(boo)
        rev2.drop(columns=[3],inplace=True)
        cols=np.append(cols,float(rev2.iloc[0,x])) #Extract column list ; x
        data.iloc[:,i]=pd.to_numeric(rev2.iloc[:,z]) # Z-data
        
    data.columns=cols
    data.index=idx
    
    if X_label != '':
        data.columns.name = X_label
    
    if Y_label != '':
        data.index.name = Y_label
    
    if dataName != '':
        data.name = dataName
    
    return data
