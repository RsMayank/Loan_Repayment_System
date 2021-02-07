# Loan_Repayment_System4
The function DecisionTree Clasifier taken various input mentioned below

if you have any Question Regarding this Model ,Please write me rsmayank25@gmail.com or tweet me <a href = 'https://twitter.com/rs_mayank'> Mayank Srivastava on Twitter </a>

##Raw Data
<a href = 'https://github.com/RsMayank/Loan_Repayment_System/blob/main/Decision_Tree_%20Dataset.csv'> Click here to see Raw Data (This data may be have Null Values) </a>

##Main Data 
<a href = 'https://github.com/RsMayank/Loan_Repayment_System/blob/main/dataLRS.csv'>Click Here to see main data file <b>dataLRS.csv</b></a>
#max_depth : int, default=None
  The maximum depth of the tree. If None, then nodes are expanded until
   all leaves are pure or until all leaves contain less than
   min_samples_split samples. 
   min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.
    
#If you are using jupyter notebook then use shift + TAB for help with function

Library Used:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
