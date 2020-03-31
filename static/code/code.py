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
# Correlation
dataframe.corr()
# Describe
dataframe.describe()
# Describe
dataframe.describe()
# Correlation
dataframe.corr()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Describe
dataframe.describe()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
