import pandas as pd
import numpy as np
from scipy.stats import mode # función mode, se usará para valor más frecuente en predicciones (votación mayoritaria en Bagging)

# Bagging Heterogéneo: Diferentes clases de estimadores, elegidos al azar.
def bagging_het(X_train, y_train, T, estimators, X_test):
    """
    Crea un modelo Bagging usando estimadores heterogéneos.
    En que genera una cantidad T de muestras bootstrap y
    para cada una de ellas se entrena un estimador escogido 
    en forma aleatoria desde la bolsa disponible. 
    La estimación final usando los modelos entrenados se
    realiza por mayoría de votos.

    Parámetros:
    -----------
    X_train : DataFrame
        Conjunto de entrenamiento sin columna target.
    y_train : Series
        Datos con las clases asociadas a cada obs. de X_train
    T : int
        Representa la cantidad de muestras bootstrap a generar,
        equivalente a la cantidad de estimadores a entrenar.
    estimators: list
        Lista con modelos base
    X_test : DataFrame
        Conjunto de test sobre el cual se somete el ensamble    

    Retorna:
    --------
    trained_model: list
        Lista con los estimadores entrenados

    yhat_test: np.array
        Array con cantidad de filas igual a la cantidad de 
        observaciones de X_test y T columnas cada una con
        la clasificación predicha por el estimador asignado

    yhat_out: Series
        resultado por mayoría de votos para el conjunto de test

    idx_oob: list
        Lista con los índices no repetidos de las observaciones
        excluidas en cada muestra bootstrap
    """
    # se almacenarán los modelos base entrenados
    trained_model = [] 

    # array Numpy de 0s. Tantas filas como observaciones en X_test y T columnasm una para cada modelo base
    # se almacenarán las predicciones de cada modelo en el conjunto de prueba
    yhat_test = np.zeros((X_test.shape[0], T)) 

    # almacenará los índices de las observaciones Out-Of-Bag (fuera de la bolsa)
    idx_oob = []

    # bucle principal, se ejecutará T veces (una por estimador base)
    for t in np.arange(0, T):
        # genera una muestra bootstrap: selecciona con reemplazo el mismo número de observaciones que el X_train original
        sa1 = X_train.sample(n=X_train.shape[0], replace=True)

        # calcula los índices Out-Of-Bag (OOB)        
        idx_oob = list(set(idx_oob + list(set(X_train.index)-set(sa1.index))))
        
        # selecciona aleatoriamente un estimador de la lista 'estimators'
        idx_estimator = np.random.randint(0, len(estimators))
        estimator = estimators[idx_estimator]

        # esta línea puede ser útil para depuración, para mostrar qué estimador se está usando
        #print(idx_estimator, end='; ')
        
        # entrena el estimador seleccionado en la muestra bootstrap actual 
        estimator.fit(sa1, y_train[sa1.index])
        trained_model.append(estimator)

        # realiza predicciones con el estimador entrenado sobre el conjunto de prueba
        # se asignan las predicciones de este estimador a la columna "t" del array yhat_test
        yhat_test[:,t] = estimator.predict(X_test)
    
    # una vez que se entrenan todos los estimadores y se han hecho sus predicciones, se combinan las predicciones mediante votación
    yhat_out = pd.Series(data=mode(yhat_test, axis=1)[0], name='yhat')

    # retorna los modelos entrenados, las predicciones individuales, la predicción final del ensamble y los índices OOB  
    return trained_model, yhat_test, yhat_out, idx_oob


def bagging_het_predict(X, estimators):
    yhat = np.zeros((X.shape[0], len(estimators)))

    for i, est in enumerate(estimators):
        yhat[:,i] = est.predict(X)

    return pd.Series(data=mode(yhat, axis=1)[0], name='yhat')


