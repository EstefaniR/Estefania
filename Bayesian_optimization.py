import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
import tensorflow as tf
import keras_tuner as kt
import random

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
df = df.drop(columns=['Mes','Year'])

# Prueba de la función
tr, vl, ts = train_val_test_split(df)

# Crear los datasets de entrenamiento, prueba y validación y verificar sus tamaños
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
np.random.seed(123)
random.seed(123)

def root_mean_squared_error(y_true, y_pred):
    rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred-y_true)))
    return rmse

def build_model(hp):
    N_UNITS = hp.Choice('N_UNITS', values=[32, 64, 128])
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.9, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    optimizer_name = hp.Choice('optimizer', values=['Adam', 'RMSprop', 'SGD'])
    activation = hp.Choice('activation', values=['tanh', 'linear', 'relu'])
    epochs = hp.Choice('EPOCHS', values=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300])
    batch_size = hp.Choice('BATCH_SIZE', values=[15, 25, 35, 45, 55, 65, 75, 85, 95])

    model = Sequential()
    model.add(LSTM(N_UNITS, input_shape=(x_tr_s.shape[1], x_tr_s.shape[2]), activation=activation))
    model.add(Dense(N_UNITS, activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUTPUT_LENGTH, activation='linear'))

    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=root_mean_squared_error)
    return model # Solo devolvemos el modelo

new_project_name = 'lstm_tuning_new1'  # Cambia el nombre del proyecto
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=100,
    directory='bayesian_optimization_extended',
    project_name='lstm_tuning_new1'
)

tuner.search(x_tr_s, y_tr_s, validation_data=(x_vl_s, y_vl_s)) # Ya no especificamos epochs ni batch_size aquí

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model, best_epochs, best_batch_size = tuner.get_best_models(num_models=1)[0], best_hps.get('EPOCHS'), best_hps.get('BATCH_SIZE')

print("Mejores hiperparámetros encontrados:")
print(f"N_UNITS: {best_hps.get('N_UNITS')}")
print(f"dropout_rate: {best_hps.get('dropout_rate')}")
print(f"learning_rate: {best_hps.get('learning_rate')}")
print(f"optimizer: {best_hps.get('optimizer')}")
print(f"activation: {best_hps.get('activation')}")
print(f"EPOCHS: {best_epochs}")
print(f"BATCH_SIZE: {best_batch_size}")

# Entrenar el modelo con los mejores hiperparámetros
history = best_model.fit(
    x_tr_s, y_tr_s,
    epochs=best_epochs,
    validation_data=(x_vl_s, y_vl_s),
    batch_size=best_batch_size,
    verbose=2
)

# Evaluación del modelo con los mejores hiperparámetros
rmse_tr_s = best_model.evaluate(x=x_tr_s, y=y_tr_s, verbose=0)
rmse_vl_s = best_model.evaluate(x=x_vl_s, y=y_vl_s, verbose=0)
rmse_ts_s = best_model.evaluate(x=x_ts_s, y=y_ts_s, verbose=0)

print('Comparativo desempeños (RMSE de datos normalizados) con optimización bayesiana:')
print(f'  RMSE train:\t {rmse_tr_s:.3f}')
print(f'  RMSE val:\t {rmse_vl_s:.3f}')
print(f'  RMSE test:\t {rmse_ts_s:.3f}')

# Invertir la transformación para obtener los valores originales
y_tr_pred_scaled = best_model.predict(x_tr_s, verbose=0)
y_vl_pred_scaled = best_model.predict(x_vl_s, verbose=0)
y_ts_pred_scaled = best_model.predict(x_ts_s, verbose=0)

y_tr_pred = scaler.inverse_transform(y_tr_pred_scaled)
y_vl_pred = scaler.inverse_transform(y_vl_pred_scaled)
y_ts_pred = scaler.inverse_transform(y_ts_pred_scaled)

# Calculo de RMSE en la escala original
rmse_tr = np.sqrt(np.mean(np.square(y_tr.squeeze() - y_tr_pred.squeeze())))
rmse_vl = np.sqrt(np.mean(np.square(y_vl.squeeze() - y_vl_pred.squeeze())))
rmse_ts = np.sqrt(np.mean(np.square(y_ts.squeeze() - y_ts_pred.squeeze())))

# Imprimir resultados en pantalla
print('Comparativo desempeños (RMSE en la escala original) con optimización bayesiana:')
print(f'  RMSE train:\t {rmse_tr:.3f}')
print(f'  RMSE val:\t {rmse_vl:.3f}')
print(f'  RMSE test:\t {rmse_ts:.3f}')

def predecir(x, model, scaler):
    y_pred_s = model.predict(x, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_s)
    return y_pred.flatten()

# Calcular predicciones sobre el set de prueba (en escala original)
y_ts_pred_original = predecir(x_ts_s, best_model, scaler)

N = len(y_ts_pred_original)
ndato = np.linspace(1,N,N)

# Invertir la transformación para obtener los valores originales del set de prueba
y_ts_original = scaler.inverse_transform(y_ts_s.reshape(-1, 1)).flatten()

# Calcular errores simples
errores = y_ts_original - y_ts_pred_original

# Calcular RMSE en la escala original
rmse_timestep = np.sqrt(np.mean(np.square(y_ts_original - y_ts_pred_original)))

# Crear un DataFrame con los valores reales y pronosticados (en escala original)
resultados = pd.DataFrame({
    'Valor Real': y_ts_original,
    'Pronóstico': y_ts_pred_original
})

# Calcular MAPE
mape = tf.reduce_mean(tf.abs((y_ts_original - y_ts_pred_original) / y_ts_original)) * 100
print("MAPE:", mape.numpy())

MSE = np.square(np.subtract(y_ts_original, y_ts_pred_original)).mean()

RMSE_final = math.sqrt(MSE)

print("RMSE (Escala Original):", RMSE_final)

# Guardar el DataFrame en un archivo CSV con los valores originales
resultados.to_csv('resultados_pronostico_optimizado_extended.csv', index=False)

# También puedes imprimir los primeros registros del DataFrame si lo deseas
print("\nPrimeros registros de los resultados (escala original):")
print(resultados.head())