# Libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show
# Loading Dataframe 
dataframe = pd.read_csv('train.csv')
# Describe dataframe
dataframe.describe()
# Dataframe
dataframe
# Describe columna 'droop'
dataframe['droop'].describe()
# Describe columna 'droop'
dataframe['droop'].describe()
# Drop
dataframe.drop(['filename'],axis=1,inplace=True)
df = pd.read_csv('static/csv/train.csv')
# Note: df is the DataFrame
# Example_
# import math
df['new_col'] = df[['col_0','col_1']].apply(lambda row: math.sqrt(row['col_0'] - row['col_1']), axis=1)

