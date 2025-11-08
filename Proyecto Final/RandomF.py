import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('krkopt_preprocessed.csv')

X= df.drop(columns=['Class_Encoded'])
y=df['Class_Encoded']

#Dvididro los datos en conjuntos de entrenmientos y prueba
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

# >> Precisión del modelo Random Forest: 0.9631
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