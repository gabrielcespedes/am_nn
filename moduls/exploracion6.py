import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr, chi2_contingency

def calidad_datos(datos):
  tipos = pd.DataFrame(datos.dtypes, columns = ['tipo'])
  nan = pd.DataFrame(datos.isna().sum(), columns = ['nan'])
  nan_prop = pd.DataFrame(datos.isna().sum()/datos.shape[0]*100, columns = ['porcentaje_nan'])  
  ceros = pd.DataFrame([datos.loc[datos[col] == 0, col].shape[0]  for col in datos.columns], columns = ['ceros'], \
                      index = datos.columns)
  ceros_prop = pd.DataFrame([datos.loc[datos[col] == 0, col].shape[0]/datos.shape[0]*100  for col in datos.columns], \
                            columns = ['porcentaje_ceros'], index = datos.columns)
  
  resumen = datos.describe(include = 'all').T
  resumen['range'] = resumen['max'] - resumen['min']
  resumen['IQR'] = resumen['75%'] - resumen['25%']
  resumen['lim_inf'] = resumen['25%'] - resumen['IQR']*1.5
  resumen['lim_sup'] = resumen['75%'] + resumen['IQR']*1.5

  resumen['atipicos'] = datos.apply(lambda x: sum(np.where((x < resumen['lim_inf'][x.name]) | \
                                  (x > resumen['lim_sup'][x.name]), 1, 0)) if x.name in resumen['lim_inf'].dropna().index else 0)
  
  return pd.concat([tipos, nan, nan_prop, ceros, ceros_prop, resumen], axis = 1).sort_values('tipo')

def graficos(calidad, datos, cols):
  num_cols = len(cols)
  num_rows = (num_cols + 2) // 3
  plt.figure(figsize = (15.1, 3.1 * num_rows)) 
  for n, i in enumerate(cols):
    plt.subplot(num_rows, 3, n+1)    
    if calidad.loc[i, 'tipo'] == 'object' or datos[i].unique().shape[0] <= 15:
      sns.countplot(y = datos[i], order = datos[i].value_counts().iloc[:16].index)
      plt.title(f'Frecuencias para {i}')
      plt.tight_layout()
    else:
      sns.histplot(datos[i], kde = True, color = 'salmon')
      plt.title(f'Distribución para {i}')
      plt.tight_layout()
    
def no_atipicos(columna):
  q1 = columna.quantile(0.25)
  q3 = columna.quantile(0.75)
  rango_iq = q3 - q1
  lim_inf = q1 - 1.5*rango_iq
  lim_sup = q3 + 1.5*rango_iq
  condicion = (columna >= lim_inf) & (columna <= lim_sup)
  return condicion

def correlacion(datos, target, dicotomic = False):
  numeric_features = datos.select_dtypes(include=[np.number]).columns.tolist()
  for feature in numeric_features:
    dfi = datos.loc[:, [feature, target]].dropna()
    corr, _ = pearsonr(dfi[feature], dfi[target])
    if isinstance(corr, np.ndarray):
      print(f"Correlación de Pearson entre {feature} y {target}: {corr[0]:.2f}")
    else:
      print(f"Correlación de Pearson entre {feature} y {target}: {corr:.2f}")

  correlation_matrix_numeric = datos[numeric_features].corr()

  plt.rcParams.update({'font.size': 12})
  plt.figure(figsize = (4,8))
  sns.heatmap(correlation_matrix_numeric.loc[:, [target]].sort_values(target, ascending = False).iloc[1:], cmap = 'Reds', annot = True)     
  
  categorical_features = datos.select_dtypes(include=[object]).columns.tolist()
  if dicotomic == True:
    for feature in categorical_features:
      target_modificado = datos[target].replace({1: 'Positiva', 0: 'Negativa'})
      chi2, p, _, _ = chi2_contingency(pd.crosstab(datos[feature], target_modificado))
      if p < 0.05:
        print(f"Chi-cuadrado entre {feature} y {target}: {chi2:.2f}, p-value: {p:.4f}. Hay relación entre las variables.")
      else:
        print(f"Chi-cuadrado entre {feature} y {target}: {chi2:.2f}, p-value: {p:.4f}. Las variables son independientes.")
  return numeric_features, categorical_features

def matriz_correlacion(datos, cols):
  df_corr = datos[cols]
  df_corr = df_corr.corr()

  fig, ax = plt.subplots(figsize = (9, 5))
  ax.set_title("Matriz de correlación para datos continuos\n")
  sns.heatmap(df_corr, annot = True, fmt = ".1f", linewidths = 0.5, ax = ax, cmap = "Blues")

def boxplots(datos, cols):
  num_cols = len(datos[cols].columns)
  num_rows_boxplot = (num_cols + 2) // 4
  plt.figure(figsize = (15, 4 * num_rows_boxplot))

  for n, columna in enumerate(datos[cols].columns):
    plt.subplot(num_rows_boxplot, 4, n+1)
    sns.boxplot(y = datos[columna])
    plt.title(f"Distribución de {columna}")
    plt.tight_layout()

  plt.show()