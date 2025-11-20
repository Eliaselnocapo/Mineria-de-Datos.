import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('krkopt_preprocessed.csv')

X = df.drop(columns=['Class_Encoded'])
y = df['Class_Encoded']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear y entrenar el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo Random Forest: {accuracy:.4f}")

# Imprimir el reporte de clasificación detallado
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Obtener la importancia de cada característica
importances = rf_model.feature_importances_
feature_names = X.columns

# Crear un DataFrame para una mejor visualización
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Importancia de cada Característica:")
print(feature_importance_df)


# ============================================================
# FUNCIONES DE RECONOCIMIENTO DE PATRONES
# ============================================================

def add_strategic_features(data):
    """
    Agrega características estratégicas de ajedrez para mejor reconocimiento de patrones
    """
    features_df = data.copy()
    
    # Distancia entre piezas
    features_df['WK_WR_distance'] = np.sqrt(
        (features_df['WK_file'] - features_df['WR_file'])**2 + 
        (features_df['WK_rank'] - features_df['WR_rank'])**2
    )
    features_df['WK_BK_distance'] = np.sqrt(
        (features_df['WK_file'] - features_df['BK_file'])**2 + 
        (features_df['WK_rank'] - features_df['BK_rank'])**2
    )
    features_df['WR_BK_distance'] = np.sqrt(
        (features_df['WR_file'] - features_df['BK_file'])**2 + 
        (features_df['WR_rank'] - features_df['BK_rank'])**2
    )
    
    # Características de confinamiento del rey
    features_df['BK_edge_distance'] = np.minimum(
        np.minimum(features_df['BK_file']-1, 8-features_df['BK_file']),
        np.minimum(features_df['BK_rank']-1, 8-features_df['BK_rank'])
    )
    
    # Torre cortando al rey
    features_df['rook_cuts_file'] = (
        (features_df['WR_file'] > features_df['BK_file']) & 
        (features_df['WR_file'] < features_df['WK_file'])
    ) | (
        (features_df['WR_file'] < features_df['BK_file']) & 
        (features_df['WR_file'] > features_df['WK_file'])
    )
    
    features_df['rook_cuts_rank'] = (
        (features_df['WR_rank'] > features_df['BK_rank']) & 
        (features_df['WR_rank'] < features_df['WK_rank'])
    ) | (
        (features_df['WR_rank'] < features_df['BK_rank']) & 
        (features_df['WR_rank'] > features_df['WK_rank'])
    )
    
    # Oposición de reyes
    features_df['kings_opposition'] = (
        (abs(features_df['WK_file'] - features_df['BK_file']) <= 1) & 
        (abs(features_df['WK_rank'] - features_df['BK_rank']) <= 1)
    )
    
    return features_df


def identify_winning_patterns(X_data, y_data, min_support=0.1):
    """
    Identifica patrones comunes en posiciones ganadoras
    """
    print("\n" + "="*60)
    print("IDENTIFICACIÓN DE PATRONES GANADORES")
    print("="*60)
    
    # Agregar características estratégicas
    data_with_features = add_strategic_features(X_data)
    data_with_features['outcome'] = y_data
    
    # Separar por clase de resultado
    patterns = {}
    
    for outcome in range(-1, 17):  # -1 es empate, 0-16 son movimientos para ganar
        outcome_data = data_with_features[data_with_features['outcome'] == outcome]
        
        if len(outcome_data) < 100:  # Saltar si hay muy pocos ejemplos
            continue
        
        pattern_dict = {
            'count': len(outcome_data),
            'avg_BK_edge_distance': outcome_data['BK_edge_distance'].mean(),
            'avg_WK_BK_distance': outcome_data['WK_BK_distance'].mean(),
            'avg_WR_BK_distance': outcome_data['WR_BK_distance'].mean(),
            'rook_cuts_file_pct': outcome_data['rook_cuts_file'].mean() * 100,
            'rook_cuts_rank_pct': outcome_data['rook_cuts_rank'].mean() * 100,
            'kings_opposition_pct': outcome_data['kings_opposition'].mean() * 100,
        }
        
        if outcome == -1:
            patterns['Empate'] = pattern_dict
        else:
            patterns[f'Victoria en {outcome} movimientos'] = pattern_dict
    
    # Mostrar patrones para victorias rápidas
    print("\nPatrones para victorias rápidas (≤5 movimientos):")
    print("-" * 50)
    for outcome, pattern in patterns.items():
        if 'Victoria' in outcome:
            moves = int(outcome.split()[2])
            if moves <= 5:
                print(f"\n{outcome}:")
                print(f"  • Total de posiciones: {pattern['count']}")
                print(f"  • Distancia promedio del Rey Negro al borde: {pattern['avg_BK_edge_distance']:.2f}")
                print(f"  • Distancia promedio Rey-Rey: {pattern['avg_WK_BK_distance']:.2f}")
                print(f"  • Distancia promedio Torre-Rey Negro: {pattern['avg_WR_BK_distance']:.2f}")
                print(f"  • Torre corta columna: {pattern['rook_cuts_file_pct']:.1f}%")
                print(f"  • Torre corta fila: {pattern['rook_cuts_rank_pct']:.1f}%")
                print(f"  • Reyes en oposición: {pattern['kings_opposition_pct']:.1f}%")
    
    return patterns


def find_position_clusters(X_data, y_data, n_clusters=5):
    """
    Encuentra grupos de posiciones similares usando K-means
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE AGRUPAMIENTO DE POSICIONES")
    print("="*60)
    
    # Enfocarse en posiciones ganadoras
    winning_mask = y_data >= 0  # Excluir empates
    winning_positions = X_data[winning_mask]
    winning_outcomes = y_data[winning_mask]
    
    if len(winning_positions) == 0:
        print("No hay suficientes posiciones ganadoras para agrupar")
        return []
    
    # Realizar agrupamiento
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(winning_positions)
    
    # Analizar grupos
    cluster_analysis = []
    for cluster_id in range(n_clusters):
        cluster_mask = clusters == cluster_id
        cluster_outcomes = winning_outcomes[cluster_mask]
        
        if len(cluster_outcomes) > 0:
            analysis = {
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'avg_moves_to_win': cluster_outcomes.mean(),
                'min_moves_to_win': cluster_outcomes.min(),
                'max_moves_to_win': cluster_outcomes.max(),
                'std_moves_to_win': cluster_outcomes.std(),
                'center': kmeans.cluster_centers_[cluster_id]
            }
            cluster_analysis.append(analysis)
    
    # Mostrar información de los grupos
    for cluster in cluster_analysis:
        print(f"\nGrupo {cluster['cluster_id']}:")
        print(f"  • Tamaño: {cluster['size']} posiciones")
        print(f"  • Promedio movimientos para ganar: {cluster['avg_moves_to_win']:.1f}")
        print(f"  • Rango: {cluster['min_moves_to_win']}-{cluster['max_moves_to_win']} movimientos")
        print(f"  • Desviación estándar: {cluster['std_moves_to_win']:.2f}")
        print(f"  • Centro del grupo (aproximado):")
        print(f"    - Rey Blanco: ({cluster['center'][0]:.1f}, {cluster['center'][1]:.1f})")
        print(f"    - Torre Blanca: ({cluster['center'][2]:.1f}, {cluster['center'][3]:.1f})")
        print(f"    - Rey Negro: ({cluster['center'][4]:.1f}, {cluster['center'][5]:.1f})")
    
    return cluster_analysis


def analyze_critical_positions(X_test, y_test, y_pred):
    """
    Identifica posiciones críticas donde el modelo comete errores
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE POSICIONES CRÍTICAS")
    print("="*60)
    
    # Encontrar posiciones mal clasificadas
    misclassified_mask = y_pred != y_test
    misclassified_X = X_test[misclassified_mask]
    misclassified_true = y_test[misclassified_mask]
    misclassified_pred = y_pred[misclassified_mask]
    
    print(f"\nTotal de posiciones mal clasificadas: {misclassified_mask.sum()}")
    print(f"Tasa de error: {misclassified_mask.sum() / len(y_test) * 100:.2f}%")
    
    # Analizar patrones de error
    error_analysis = defaultdict(int)
    for true_val, pred_val in zip(misclassified_true, misclassified_pred):
        error_diff = abs(true_val - pred_val)
        if true_val == -1:
            error_type = f"Empate predicho como victoria en {pred_val}"
        elif pred_val == -1:
            error_type = f"Victoria en {true_val} predicha como empate"
        else:
            error_type = f"Diferencia de {error_diff} movimientos"
        error_analysis[error_type] += 1
    
    # Ordenar por frecuencia
    sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)
    
    print("\nErrores de clasificación más comunes:")
    for i, (error_type, count) in enumerate(sorted_errors[:10], 1):
        print(f"  {i}. {error_type}: {count} casos")
    
    # Analizar características de posiciones mal clasificadas
    if len(misclassified_X) > 0:
        misclassified_features = add_strategic_features(misclassified_X)
        print("\nCaracterísticas promedio de posiciones mal clasificadas:")
        print(f"  • Distancia Rey Negro al borde: {misclassified_features['BK_edge_distance'].mean():.2f}")
        print(f"  • Distancia Rey-Rey: {misclassified_features['WK_BK_distance'].mean():.2f}")
        print(f"  • Torre corta al rey: {misclassified_features['rook_cuts_file'].mean()*100:.1f}%")
    
    return misclassified_X, misclassified_true, misclassified_pred


def generate_strategic_recommendations(position, model):
    """
    Genera recomendaciones estratégicas para una posición dada
    
    Args:
        position: lista de 6 valores [WK_file, WK_rank, WR_file, WR_rank, BK_file, BK_rank]
        model: modelo Random Forest entrenado
    """
    # Predecir resultado
    position_array = np.array(position).reshape(1, -1)
    prediction = model.predict(position_array)[0]
    probabilities = model.predict_proba(position_array)[0]
    
    # Crear DataFrame para calcular características
    pos_df = pd.DataFrame([position], columns=X.columns)
    pos_with_features = add_strategic_features(pos_df)
    
    recommendations = []
    
    # Analizar posición
    if pos_with_features['BK_edge_distance'].values[0] < 2:
        recommendations.append("Rey Negro cerca del borde - bueno para confinamiento")
    
    if pos_with_features['rook_cuts_file'].values[0]:
        recommendations.append("Torre cortando al Rey Negro en columna")
    
    if pos_with_features['rook_cuts_rank'].values[0]:
        recommendations.append("Torre cortando al Rey Negro en fila")
    
    if pos_with_features['WK_BK_distance'].values[0] < 3:
        recommendations.append("Reyes cercanos - mantener presión")
    elif pos_with_features['WK_BK_distance'].values[0] > 5:
        recommendations.append("Reyes distantes - considerar acercar el Rey Blanco")
    
    if pos_with_features['WR_BK_distance'].values[0] < 2:
        recommendations.append("Torre muy cerca del Rey Negro - cuidado con ahogado")
    
    # Resultado predicho
    if prediction == -1:
        recommendations.append("Posición probablemente tablas")
        strategy = "Buscar mejora en la posición de las piezas"
    elif prediction <= 3:
        recommendations.append(f"Victoria rápida posible en {int(prediction)} movimientos")
        strategy = "Mantener presión y evitar errores"
    elif prediction <= 8:
        recommendations.append(f"Victoria en {int(prediction)} movimientos")
        strategy = "Jugar con precisión para convertir la ventaja"
    else:
        recommendations.append(f"Victoria requiere {int(prediction)} movimientos")
        strategy = "Paciencia necesaria, mejorar posición gradualmente"
    
    return {
        'prediction': prediction,
        'confidence': max(probabilities),
        'recommendations': recommendations,
        'strategy': strategy
    }


def analyze_specific_position(position_str, model):
    """
    Analiza una posición específica de ajedrez
    
    Args:
        position_str: string en formato "WKf,WKr,WRf,WRr,BKf,BKr" 
                     donde f=columna (a-h o 1-8), r=fila (1-8)
        model: modelo Random Forest entrenado
    """
    print("\n" + "-"*50)
    print(f"Analizando posición: {position_str}")
    print("-"*50)
    
    # Parsear posición
    parts = position_str.split(',')
    
    # Convertir letras a números si es necesario
    file_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
    
    position = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # columnas (files)
            if part in file_map:
                position.append(file_map[part])
            else:
                position.append(int(part))
        else:  # filas (ranks)
            position.append(int(part))
    
    # Obtener recomendaciones
    analysis = generate_strategic_recommendations(position, model)
    
    # Mostrar resultados
    print(f"Rey Blanco: ({parts[0]},{parts[1]})")
    print(f"Torre Blanca: ({parts[2]},{parts[3]})")
    print(f"Rey Negro: ({parts[4]},{parts[5]})")
    print(f"\nResultado predicho: ", end="")
    
    if analysis['prediction'] == -1:
        print("TABLAS")
    else:
        print(f"VICTORIA EN {int(analysis['prediction'])} MOVIMIENTOS")
    
    print(f"Confianza: {analysis['confidence']:.2%}")
    print(f"\nEstrategia recomendada: {analysis['strategy']}")
    print("\nAnálisis estratégico:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    return analysis


def visualize_patterns(X_data, y_data):
    """
    Visualiza los patrones encontrados en los datos
    """
    print("\n" + "="*60)
    print("VISUALIZACIÓN DE PATRONES")
    print("="*60)
    
    # Agregar características estratégicas
    data_with_features = add_strategic_features(X_data)
    data_with_features['outcome'] = y_data
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribución de resultados
    outcome_counts = data_with_features['outcome'].value_counts().sort_index()
    axes[0, 0].bar(outcome_counts.index, outcome_counts.values)
    axes[0, 0].set_xlabel('Resultado (movimientos para ganar, -1=tablas)')
    axes[0, 0].set_ylabel('Número de posiciones')
    axes[0, 0].set_title('Distribución de Resultados')
    
    # 2. Relación entre distancia del rey al borde y resultado
    winning_data = data_with_features[data_with_features['outcome'] >= 0]
    if len(winning_data) > 0:
        axes[0, 1].scatter(winning_data['BK_edge_distance'], 
                          winning_data['outcome'], 
                          alpha=0.3, s=1)
        axes[0, 1].set_xlabel('Distancia del Rey Negro al borde')
        axes[0, 1].set_ylabel('Movimientos para ganar')
        axes[0, 1].set_title('Rey Negro al borde vs Movimientos para ganar')
    
    # 3. Distancia entre reyes vs resultado
    if len(winning_data) > 0:
        axes[1, 0].scatter(winning_data['WK_BK_distance'], 
                          winning_data['outcome'], 
                          alpha=0.3, s=1)
        axes[1, 0].set_xlabel('Distancia entre Reyes')
        axes[1, 0].set_ylabel('Movimientos para ganar')
        axes[1, 0].set_title('Distancia Rey-Rey vs Movimientos para ganar')
    
    # 4. Importancia de características
    axes[1, 1].barh(feature_importance_df['Feature'][::-1], 
                    feature_importance_df['Importance'][::-1])
    axes[1, 1].set_xlabel('Importancia')
    axes[1, 1].set_title('Importancia de Características')
    
    plt.tight_layout()
    plt.savefig('chess_patterns_visualization.png', dpi=150)
    plt.show()
    
    print("Visualización guardada como 'chess_patterns_visualization.png'")


# ============================================================
# EJECUCIÓN DE ANÁLISIS DE PATRONES
# ============================================================

print("\n" + "="*70)
print(" ANÁLISIS DE RECONOCIMIENTO DE PATRONES EN AJEDREZ")
print("="*70)

# 1. Identificar patrones ganadores
patterns = identify_winning_patterns(X_train, y_train)

# 2. Encontrar grupos de posiciones
clusters = find_position_clusters(X_train, y_train, n_clusters=5)

# 3. Analizar posiciones críticas
critical_positions = analyze_critical_positions(X_test, y_test, y_pred)

# 4. Ejemplos de análisis de posiciones específicas
print("\n" + "="*60)
print("EJEMPLOS DE ANÁLISIS DE POSICIONES")
print("="*60)

# Ejemplo 1: Posición clásica ganadora
example1 = analyze_specific_position("d,4,h,4,d,8", rf_model)

# Ejemplo 2: Torre cortando al rey
example2 = analyze_specific_position("e,6,e,4,h,8", rf_model)

# Ejemplo 3: Reyes cercanos
example3 = analyze_specific_position("f,6,h,5,f,8", rf_model)

# 5. Visualizar patrones (opcional - comentar si no se desea)
# visualize_patterns(X_train, y_train)

print("\n" + "="*70)
print(" ANÁLISIS COMPLETO")
print("="*70)
print(f"Precisión del modelo: {accuracy:.4f}")
print("\nPerspectivas clave:")
print("1. La distancia del Rey Negro al borde es crucial para victorias rápidas")
print("2. Torre cortando al Rey es un patrón estratégico clave")
print("3. Distancia cercana Rey-Rey correlaciona con victorias más rápidas")
print("4. Diferentes grupos de posiciones muestran patrones de victoria distintos")