import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARTE 1: MODELO RANDOM FOREST (CÓDIGO ORIGINAL)
# ============================================================

print("\n" + "="*60)
print(" 1. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO")
print("="*60)

try:
    df = pd.read_csv('krkopt_preprocessed.csv')
except FileNotFoundError:
    print("Error: No se encuentra 'krkopt_preprocessed.csv'. Ejecuta preppros.py primero.")
    exit()

X = df.drop(columns=['Class_Encoded'])
y = df['Class_Encoded']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear y entrenar el modelo Random Forest
print("... Entrenando Random Forest (n_estimators=100) ...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = rf_model.predict(X_test)

# --- RESULTADOS ORIGINALES ---

# 1. Precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"\n>> Precisión del modelo Random Forest: {accuracy:.4f}")

# 2. Reporte de Clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))

# 3. Importancia de Características
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Importancia de cada Característica:")
print(feature_importance_df)


# ============================================================
# PARTE 2: MINERÍA DE PATRONES DE EMPATE (NUEVO AGREGADO)
# ============================================================

def calculate_chess_metrics(data):
    """Calcula métricas geométricas esenciales para el ajedrez."""
    df_calc = data.copy()
    
    # Distancia Euclidiana entre Reyes (Raíz cuadrada de la suma de los cuadrados)
    df_calc['Dist_Reyes'] = np.sqrt(
        (df_calc['WK_file'] - df_calc['BK_file'])**2 + 
        (df_calc['WK_rank'] - df_calc['BK_rank'])**2
    )
    
    # Distancia de la Torre al Rey Negro
    df_calc['Dist_Torre_ReyNegro'] = np.sqrt(
        (df_calc['WR_file'] - df_calc['BK_file'])**2 + 
        (df_calc['WR_rank'] - df_calc['BK_rank'])**2
    )
    
    # Distancia del Rey Negro al Borde más cercano (0 = está en el borde)
    # Los valores de file/rank van de 1 a 8.
    # Mínimo entre (distancia a la izquierda, distancia a la derecha, distancia abajo, distancia arriba)
    df_calc['ReyNegro_Dist_Borde'] = np.minimum(
        np.minimum(df_calc['BK_file'] - 1, 8 - df_calc['BK_file']),
        np.minimum(df_calc['BK_rank'] - 1, 8 - df_calc['BK_rank'])
    )
    
    # Regla Lógica: ¿Está el Rey Negro en el borde? (True/False)
    df_calc['Patron_Rey_Borde'] = df_calc['ReyNegro_Dist_Borde'] == 0
    
    # Regla Lógica: ¿Reyes en Oposición cercana? (Distancia < 2.0)
    df_calc['Patron_Reyes_Cerca'] = df_calc['Dist_Reyes'] < 2.0
    
    return df_calc

def mine_draw_patterns(X_original, y_original):
    print("\n" + "="*60)
    print(" 2. MINERÍA DE DATOS: PATRONES DE EMPATE (DRAW)")
    print("="*60)
    
    # Unimos X e y para analizar patrones sobre el resultado real
    full_data = X_original.copy()
    full_data['Clase'] = y_original
    
    # Calculamos las métricas geométricas
    full_data = calculate_chess_metrics(full_data)
    
    # Filtramos: Empates (-1) vs No Empates (Cualquier otro valor)
    empates = full_data[full_data['Clase'] == -1]
    no_empates = full_data[full_data['Clase'] != -1]
    
    n_empates = len(empates)
    
    if n_empates == 0:
        print("No hay empates en el set de prueba para analizar.")
        return

    print(f"Total de partidas analizadas: {len(full_data)}")
    print(f"Total de Empates encontrados: {n_empates} ({n_empates/len(full_data):.1%})")
    
    # --- A. COMPARATIVA DE PROMEDIOS (Contrast Mining) ---
    # Esto nos dice qué diferencia numérica hay entre una partida ganada y un empate
    print("\n[A] DIFERENCIAS ESTRUCTURALES (Promedios)")
    print(f"{'Métrica':<30} | {'En Empates':<12} | {'En Victorias':<12} | {'Diferencia'}")
    print("-" * 70)
    
    metrics = {
        'ReyNegro_Dist_Borde': 'Distancia Rey Negro a Borde',
        'Dist_Reyes': 'Distancia entre Reyes',
        'Dist_Torre_ReyNegro': 'Distancia Torre-Rey N.'
    }
    
    for col, name in metrics.items():
        mean_draw = empates[col].mean()
        mean_win = no_empates[col].mean()
        diff = mean_draw - mean_win
        print(f"{name:<30} | {mean_draw:>10.2f}   | {mean_win:>10.2f}   | {diff:>+10.2f}")

    print("\n  -> INTERPRETACIÓN:")
    print("     * Si 'Distancia a Borde' es menor en empates, indica patrones de Ahogado.")
    print("     * Si 'Distancia Torre-Rey N.' es menor en empates, indica riesgo de captura de la torre.")

    # --- B. FRECUENCIA DE REGLAS LÓGICAS (Rule Mining) ---
    print("\n[B] REGLAS LÓGICAS MÁS FRECUENTES EN EMPATES")
    
    # Regla 1: Rey en Borde
    count_borde = empates['Patron_Rey_Borde'].sum()
    pct_borde = (count_borde / n_empates) * 100
    
    # Regla 2: Reyes Cerca
    count_cerca = empates['Patron_Reyes_Cerca'].sum()
    pct_cerca = (count_cerca / n_empates) * 100
    
    print(f"\n1. PATRÓN DE AHOGADO (STALEMATE):")
    print(f"   - Condición: El Rey Negro está en el borde del tablero.")
    print(f"   - Frecuencia: Ocurre en el {pct_borde:.1f}% de los empates.")
    
    print(f"\n2. PATRÓN DE DEFENSA ACTIVA:")
    print(f"   - Condición: Los Reyes están muy cerca (distancia < 2).")
    print(f"   - Frecuencia: Ocurre en el {pct_cerca:.1f}% de los empates.")

# Ejecutamos la nueva función usando el conjunto de PRUEBA (para ser justos con la evaluación)
mine_draw_patterns(X_test, y_test)

print("\n" + "="*60)
print(" FIN DEL PROCESO")
print("="*60)