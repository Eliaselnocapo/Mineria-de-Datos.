import pandas as pd
import numpy as np

# Cargar los datos del archivo, sin encabezado
df = pd.read_csv('chess+king+rook+vs+king/krkopt.data', header=None)
# Asignar nombres de columna según el archivo krkopt.info
columns = [
    'WK_file', 'WK_rank', 'WR_file', 'WR_rank',
    'BK_file', 'BK_rank', 'Class'
]
df.columns = columns

# --- 1. Mapear archivos (letras) a rangos (números) ---
# Mapeo de a-h a 1-8
file_mapping = {file: i + 1 for i, file in enumerate('abcdefgh')}

# Aplicar el mapeo a las columnas de archivo
for col in ['WK_file', 'WR_file', 'BK_file']:
    df[col] = df[col].map(file_mapping)

# --- 2. Codificar la variable de clase (el resultado del juego) ---
class_mapping = {
    'draw': -1,
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16
}

df['Class_Encoded'] = df['Class'].map(class_mapping)

# Eliminar la columna de clase original (categórica)
df = df.drop(columns=['Class'])

# Guardar los datos preprocesados en un nuevo archivo CSV
df.to_csv('krkopt_preprocessed.csv', index=False)