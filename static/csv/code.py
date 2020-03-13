# Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('regresion_lineal.csv')
# Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('regresion_logistica.csv')
# Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('train.csv')
# Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('regresion_logistica.csv')
# Revisando si tiene NaN la columna 'Age'
dataframe['Age'].isnull().sum()
# Describe columna 'Age'
dataframe['Age'].describe()
# Imputando valor a los valores NaN de la columna 'Age'
dataframe['Age'].fillna('28', inplace=True)
