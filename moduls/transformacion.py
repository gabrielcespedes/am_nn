from sklearn.preprocessing import StandardScaler
import pandas as pd

def escalado(datos, columnas):
    cols = columnas
    data_to_scale = datos[cols]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data_to_scale)
    scaled_df = pd.DataFrame(scaled_data, columns = cols, index = datos.index)

    datos[cols] = scaled_df
    return datos

