# Machine Learning for developers "ML4D"

## Versión actual

version 0.77

## Descripción

Herramienta web para realizar Machine learning que permite:

1. Subir un arhivo de tipo csv.
2. Ver estadística básica del dataset.
3. Transformar el archivo (Renombrar, Imputar, etc.).
4. Aplicar modelos de regresión y clasificación al dataset.
5. Realizar un deploy para realizar predicciones directamente en la herramienta.
6. Realizar el deploy de un API REST para realizar predicciones con formato JSON.

## Librerías

web.py==0.40
pandas==1.0.1
numpy==1.18.1
statsmodels==0.11.1
scipy==1.4.1
matplotlib==3.1.3
seaborn==0.10.0
sklearn==0.0
nbformat==5.0.4

## Cambios en las versiones

### Versión v0.77

1. Deploy y API percistente una vez que se entrena algún modelo.
2. Menú actualizado.

### Versión v0.76

1. Deploy y API para un modelo entrenado
   
### Versión v0.69

2. Dataset en csv default
3. Númera las columnas
4. Quita columnas
5. Listado de columnas
6. Columnas duplicadas
7. Número de valores NaN
8. Gráfica countplot por cada columna
9.  5 primeros valores por columna
10. Imputa valores a los Nan
11. Reemplaza valores
12. Muestra la moda de cada columna
13. Muestra la mediana de cada columna
14. Muestra la media de cada columna
15. Acerca de
16. Protección de static/