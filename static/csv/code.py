#Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('Leads.csv')# Describe 'Lead Origin
