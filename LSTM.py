import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import RMSprop,Adam, SGD
import tensorflow as tf


def train_val_test_split(dataframe, tr_size=0.8, vl_size=0.1):
    N = dataframe.shape[0]
    Ntrain = int(tr_size * N)
    Nval = int(vl_size * N)

    train = dataframe[0:Ntrain]
    val = dataframe[Ntrain:Ntrain+Nval]
    test = dataframe[Ntrain+Nval:]

    return train, val, test

def crear_dataset_supervisado(array, input_length, output_length):


    X, Y = [], []
    shape = array.shape
    if len(shape) == 1:
        fils, cols = array.shape[0], 1
        array = array.reshape(fils, cols)
    else:
        fils, cols = array.shape

    for i in range(fils - input_length - output_length+1):
        X.append(array[i:i + input_length, 0:cols-1])
        Y.append(array[i + input_length:i + input_length + output_length, 9].reshape(output_length, 1))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


# Cargar los datos
df = pd.read_csv("Outrnn.csv", sep=",")
#La siguiente instrucción sirve para crear una columna con los índices de cada uno de los renglones.
#df = df.reset_index()
df = df.drop(columns=['Mes','Year'])

# Prueba de la función
tr, vl, ts = train_val_test_split(df)

# Datasets de entrenamiento, prueba y validación
INPUT_LENGTH = 2
OUTPUT_LENGTH = 1

x_tr, y_tr = crear_dataset_supervisado(tr.values, INPUT_LENGTH, OUTPUT_LENGTH)
x_vl, y_vl = crear_dataset_supervisado(vl.values, INPUT_LENGTH, OUTPUT_LENGTH)
x_ts, y_ts = crear_dataset_supervisado(ts.values, INPUT_LENGTH, OUTPUT_LENGTH)

# Imprimir información en pantalla
print('Tamaños entrada (BATCHES x INPUT_LENGTH x FEATURES) y de salida (BATCHES x OUTPUT_LENGTH x FEATURES)')
print(f'Set de entrenamiento - x_tr: {x_tr.shape}, y_tr: {y_tr.shape}')
print(f'Set de validación - x_vl: {x_vl.shape}, y_vl: {y_vl.shape}')
print(f'Set de prueba - x_ts: {x_ts.shape}, y_ts: {y_ts.shape}')


def escalar_dataset(data_input):
    # col_ref = df.columns.get_loc(col_ref)
    NFEATS = data_input['x_tr'].shape[2]

    scalers = [MinMaxScaler(feature_range=(-1, 1)) for i in range(NFEATS)]

    x_tr_s = np.zeros(data_input['x_tr'].shape)
    x_vl_s = np.zeros(data_input['x_vl'].shape)
    x_ts_s = np.zeros(data_input['x_ts'].shape)
    y_tr_s = np.zeros(data_input['y_tr'].shape)
    y_vl_s = np.zeros(data_input['y_vl'].shape)
    y_ts_s = np.zeros(data_input['y_ts'].shape)

    for i in range(NFEATS):
        x_tr_s[:, :, i] = scalers[i].fit_transform(x_tr[:, :, i])
        x_vl_s[:, :, i] = scalers[i].transform(x_vl[:, :, i])
        x_ts_s[:, :, i] = scalers[i].transform(x_ts[:, :, i])

    scalerY= MinMaxScaler(feature_range=(-1, 1))

    y_tr_s[:, :, 0] = scalerY.fit_transform(y_tr[:, :, 0])
    y_vl_s[:, :, 0] = scalerY.transform(y_vl[:, :, 0])
    y_ts_s[:, :, 0] = scalerY.transform(y_ts[:, :, 0])

    data_scaled = {
        'x_tr_s': x_tr_s, 'y_tr_s': y_tr_s,
        'x_vl_s': x_vl_s, 'y_vl_s': y_vl_s,
        'x_ts_s': x_ts_s, 'y_ts_s': y_ts_s,
    }

    return data_scaled, scalerY

# Escalamiento del dataset con la función anterior
data_in = {
    'x_tr': x_tr, 'y_tr': y_tr,
    'x_vl': x_vl, 'y_vl': y_vl,
    'x_ts': x_ts, 'y_ts': y_ts,
}

data_s, scaler = escalar_dataset(data_in)

x_tr_s, y_tr_s = data_s['x_tr_s'], data_s['y_tr_s']
x_vl_s, y_vl_s = data_s['x_vl_s'], data_s['y_vl_s']
x_ts_s, y_ts_s = data_s['x_ts_s'], data_s['y_ts_s']


tf.keras.utils.set_random_seed(123)

# El modelo
N_UNITS = 64 # estado oculto (h) y de la celdad de memoria (c) (128)
INPUT_SHAPE = (x_tr_s.shape[1], x_tr_s.shape[2]) # 24 (horas) x 13 (features)

# Crear el modelo
modelo = Sequential()
modelo.add((LSTM(N_UNITS, input_shape=INPUT_SHAPE)))
modelo.add(Dense(N_UNITS, activation="linear"))
modelo.add(Dropout(0.3))
modelo.add(Dense(OUTPUT_LENGTH, activation='linear'))

def root_mean_squared_error(y_true, y_pred):
    rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred-y_true)))
    return rmse


# Compilación
optimizador = RMSprop(learning_rate=1e-4) # 5e-5
modelo.compile(
    optimizer = optimizador,
    loss = root_mean_squared_error,
)

# Entrenamiento
EPOCHS = 10
BATCH_SIZE = 45

historia = modelo.fit(

    x = x_tr_s,
    y = y_tr_s,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (x_vl_s, y_vl_s),
    verbose=2
)

# RMSE para train, val y test
rmse_tr_s = modelo.evaluate(x=x_tr_s, y=y_tr_s, verbose=0)
rmse_vl_s = modelo.evaluate(x=x_vl_s, y=y_vl_s, verbose=0)
rmse_ts_s = modelo.evaluate(x=x_ts_s, y=y_ts_s, verbose=0)

print('Comparativo desempeños (RMSE de datos normalizados):')
print(f'  RMSE train:\t {rmse_tr_s:.3f}')
print(f'  RMSE val:\t {rmse_vl_s:.3f}')
print(f'  RMSE test:\t {rmse_ts_s:.3f}')

# Invertir la transformación para obtener los valores originales
y_tr_pred = scaler.inverse_transform(modelo.predict(x_tr_s, verbose=0))
y_vl_pred = scaler.inverse_transform(modelo.predict(x_vl_s, verbose=0))
y_ts_pred = scaler.inverse_transform(modelo.predict(x_ts_s, verbose=0))

# Calculo RMSE en la escala original
rmse_tr = np.sqrt(np.mean(np.square(y_tr.squeeze() - y_tr_pred.squeeze())))
rmse_vl = np.sqrt(np.mean(np.square(y_vl.squeeze() - y_vl_pred.squeeze())))
rmse_ts = np.sqrt(np.mean(np.square(y_ts.squeeze() - y_ts_pred.squeeze())))

# Imprimir resultados
print('Comparativo desempeños (RMSE en la escala original):')
print(f'  RMSE train:\t {rmse_tr:.3f}')
print(f'  RMSE val:\t {rmse_vl:.3f}')
print(f'  RMSE test:\t {rmse_ts:.3f}')

def predecir(x, model, scaler):

    # Calcular predicción escalada en el rango de -1 a 1
    y_pred_s = model.predict(x, verbose=0)

    # Llevar la predicción a la escala original
    y_pred = scaler.inverse_transform(y_pred_s)

    return y_pred.flatten()

# Calcular pronósticos sobre el set de prueba
y_ts_pred = predecir(x_ts_s, modelo, scaler)

N = len(y_ts_pred)    # Número de predicciones
ndato = np.linspace(1,N,N)

# Obtener los valores originales
y_ts_original = scaler.inverse_transform(y_ts_s.reshape(-1, 1)).flatten()
y_ts_pred_original = scaler.inverse_transform(y_ts_pred.reshape(-1, 1)).flatten()


# Cálculo de errores simples
errores = y_ts.flatten()-y_ts_pred

# Calculo RMSE en la escala original
rmse_timestep = np.sqrt(np.mean(np.square(y_ts.flatten() - y_ts_pred.flatten())))


# DataFrame con las fechas y las predicciones
resultados = pd.DataFrame({
    'Valor Real': y_ts.flatten(),
    'Pronóstico': y_ts_pred
})

# Calcular MAPE
mape = tf.reduce_mean(tf.abs((y_ts.flatten() - y_ts_pred) / y_ts.flatten())) * 100
print("MAPE:", mape.numpy())

MSE = np.square(np.subtract(y_ts.flatten(), y_ts_pred)).mean()

RMSE = math.sqrt(MSE)

print("RMSE", RMSE)

# Guardar el DataFrame en un archivo CSV
resultados.to_csv('resultados_pronostico.csv', index=False)

# También puedes imprimir los primeros registros del DataFrame si lo deseas
print(resultados.head())