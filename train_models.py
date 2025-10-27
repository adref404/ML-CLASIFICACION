# train_models.py
# Sistema de Predicción de Enfermedades Cardíacas
# Entrenamiento y evaluación de modelos de Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ============================================================================


def load_data(filepath="heart.csv"):
    """
    Carga el dataset de enfermedades cardíacas

    Args:
        filepath: Ruta al archivo CSV

    Returns:
        DataFrame con los datos
    """
    print("[INFO] Cargando datos...")
    df = pd.read_csv(filepath)
    print(f"[OK] Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df


def explore_data(df):
    """
    Realiza exploración inicial del dataset

    Args:
        df: DataFrame con los datos
    """
    print("\n" + "=" * 60)
    print("=== EXPLORACION DE DATOS ===")
    print("=" * 60)

    print("\n[INFO] Información General:")
    print(df.info())

    print("\n[INFO] Estadísticas Descriptivas:")
    print(df.describe())

    print("\n[INFO] Valores Nulos:")
    print(df.isnull().sum())

    print("\n[INFO] Distribución de la Variable Objetivo:")
    print(df["target"].value_counts())
    print(f"\nBalance: {df['target'].value_counts(normalize=True) * 100}")

    return df


# ============================================================================
# 2. PREPROCESAMIENTO
# ============================================================================


def preprocess_data(df):
    """
    Preprocesa los datos: limpieza y normalización

    Args:
        df: DataFrame con los datos

    Returns:
        X: Features normalizadas
        y: Variable objetivo
        scaler: Objeto scaler ajustado
    """
    print("\n" + "=" * 60)
    print("=== PREPROCESAMIENTO DE DATOS ===")
    print("=" * 60)

    # Separar features y target
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"\n[OK] Features (X): {X.shape}")
    print(f"[OK] Target (y): {y.shape}")

    # Normalización de features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\n[OK] Normalización completada (StandardScaler)")

    return X_scaled, y, scaler


# ============================================================================
# 3. DIVISIÓN DE DATOS
# ============================================================================


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide datos en conjuntos de entrenamiento y prueba

    Args:
        X: Features
        y: Target
        test_size: Proporción de datos para prueba (default: 0.2)
        random_state: Semilla aleatoria

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("=== DIVISION DE DATOS ===")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"\n[INFO] Conjunto de Entrenamiento: {X_train.shape[0]} muestras ({(1-test_size)*100:.0f}%)"
    )
    print(
        f"[INFO] Conjunto de Prueba: {X_test.shape[0]} muestras ({test_size*100:.0f}%)"
    )

    print(f"\n[OK] Distribución en Train: {dict(y_train.value_counts())}")
    print(f"[OK] Distribución en Test: {dict(y_test.value_counts())}")

    return X_train, X_test, y_train, y_test


# ============================================================================
# 4. DEFINICIÓN DE MODELOS
# ============================================================================


def get_models():
    """
    Define los modelos a entrenar

    Returns:
        Diccionario con los modelos
    """
    models = {
        # MODELOS DE CAJA BLANCA (Interpretables)
        "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
        # MODELOS DE CAJA NEGRA (Complejos)
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True),
    }

    return models


# ============================================================================
# 5. ENTRENAMIENTO Y VALIDACIÓN CRUZADA
# ============================================================================


def train_with_cross_validation(models, X_train, y_train, cv=5):
    """
    Entrena modelos con validación cruzada K-Fold

    Args:
        models: Diccionario de modelos
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        cv: Número de folds (default: 5)

    Returns:
        Diccionario con modelos entrenados y scores de CV
    """
    print("\n" + "=" * 60)
    print("=== ENTRENAMIENTO CON VALIDACION CRUZADA (K=5) ===")
    print("=" * 60)

    trained_models = {}
    cv_scores = {}

    for name, model in models.items():
        print(f"\n[INFO] Entrenando: {name}")

        # Validación cruzada
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_scores[name] = scores

        print(f"   - CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"   - Scores por fold: {scores}")

        # Entrenamiento final con todos los datos de train
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   [OK] Modelo entrenado")

    return trained_models, cv_scores


# ============================================================================
# 6. EVALUACIÓN EN CONJUNTO DE PRUEBA
# ============================================================================


def evaluate_models(models, X_test, y_test):
    """
    Evalúa modelos en conjunto de prueba y calcula métricas

    Args:
        models: Diccionario de modelos entrenados
        X_test: Features de prueba
        y_test: Target de prueba

    Returns:
        DataFrame con métricas de todos los modelos
    """
    print("\n" + "=" * 60)
    print("=== EVALUACION EN CONJUNTO DE PRUEBA ===")
    print("=" * 60)

    results = []

    for name, model in models.items():
        print(f"\n[INFO] Evaluando: {name}")

        # Predicciones
        y_pred = model.predict(X_test)

        # Cálculo de métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        results.append(
            {
                "Modelo": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )

        print(f"   - Accuracy:  {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall:    {recall:.4f}")
        print(f"   - F1-Score:  {f1:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Matriz de Confusión:")
        print(f"   {cm}")

    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Accuracy", ascending=False)

    return results_df


# ============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ============================================================================


def plot_results(results_df):
    """
    Genera visualizaciones de los resultados

    Args:
        results_df: DataFrame con métricas
    """
    print("\n" + "=" * 60)
    print("=== GENERANDO VISUALIZACIONES ===")
    print("=" * 60)

    # Configuración de estilo
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Comparación de métricas
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(results_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, results_df[metric], width, label=metric)

    axes[0].set_xlabel("Modelo", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Comparación de Métricas por Modelo", fontsize=14, fontweight="bold"
    )
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(results_df["Modelo"], rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim([0.7, 1.0])

    # Gráfico 2: Heatmap de métricas
    metrics_data = results_df[metrics].T
    sns.heatmap(
        metrics_data,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        xticklabels=results_df["Modelo"],
        yticklabels=metrics,
        cbar_kws={"label": "Score"},
        ax=axes[1],
        vmin=0.7,
        vmax=1.0,
    )
    axes[1].set_title("Heatmap de Métricas", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/resultados_modelos.png", dpi=300, bbox_inches="tight")
    print("\n[OK] Gráficos guardados: outputs/resultados_modelos.png")
    plt.show()


def plot_confusion_matrices(models, X_test, y_test):
    """
    Genera matrices de confusión para todos los modelos

    Args:
        models: Diccionario de modelos entrenados
        X_test: Features de prueba
        y_test: Target de prueba
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=["Sin Enfermedad", "Con Enfermedad"],
            yticklabels=["Sin Enfermedad", "Con Enfermedad"],
        )
        axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Real")
        axes[idx].set_xlabel("Predicción")

    plt.tight_layout()
    plt.savefig("outputs/matrices_confusion.png", dpi=300, bbox_inches="tight")
    print("[OK] Matrices de confusión guardadas: outputs/matrices_confusion.png")
    plt.show()


# ============================================================================
# 8. TABLA DE RESULTADOS
# ============================================================================


def print_results_table(results_df):
    """
    Imprime tabla formateada de resultados

    Args:
        results_df: DataFrame con métricas
    """
    print("\n" + "=" * 80)
    print("=== TABLA DE RESULTADOS FINALES ===")
    print("=" * 80)
    print()
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("=" * 80)

    # Mejor modelo
    best_model = results_df.iloc[0]
    print(f"\n[MEJOR] MEJOR MODELO: {best_model['Modelo']}")
    print(f"   - Accuracy:  {best_model['Accuracy']:.2%}")
    print(f"   - Precision: {best_model['Precision']:.2%}")
    print(f"   - Recall:    {best_model['Recall']:.2%}")
    print(f"   - F1-Score:  {best_model['F1-Score']:.2%}")


# ============================================================================
# 9. GUARDAR MODELOS
# ============================================================================


def save_models(models, scaler, results_df):
    """
    Guarda modelos entrenados y resultados

    Args:
        models: Diccionario de modelos
        scaler: Objeto scaler
        results_df: DataFrame con resultados
    """
    print("\n" + "=" * 60)
    print("=== GUARDANDO MODELOS ===")
    print("=" * 60)

    # Crear carpeta outputs si no existe
    import os

    os.makedirs("outputs", exist_ok=True)

    # Guardar modelos
    for name, model in models.items():
        filename = f"outputs/model_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"[OK] {name} guardado: {filename}")

    # Guardar scaler
    joblib.dump(scaler, "outputs/scaler.pkl")
    print(f"[OK] Scaler guardado: outputs/scaler.pkl")

    # Guardar resultados
    results_df.to_csv("outputs/resultados_metricas.csv", index=False)
    print(f"[OK] Resultados guardados: outputs/resultados_metricas.csv")


# ============================================================================
# 10. FUNCIÓN PRINCIPAL
# ============================================================================


def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("=" * 80)
    print("SISTEMA DE PREDICCION DE ENFERMEDADES CARDIACAS")
    print("=" * 80)

    # 1. Cargar datos
    df = load_data("heart.csv")

    # 2. Explorar datos
    df = explore_data(df)

    # 3. Preprocesar
    X, y, scaler = preprocess_data(df)

    # 4. Dividir datos
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 5. Obtener modelos
    models = get_models()

    # 6. Entrenar con validación cruzada
    trained_models, cv_scores = train_with_cross_validation(
        models, X_train, y_train, cv=5
    )

    # 7. Evaluar modelos
    results_df = evaluate_models(trained_models, X_test, y_test)

    # 8. Visualizar resultados
    plot_results(results_df)
    plot_confusion_matrices(trained_models, X_test, y_test)

    # 9. Imprimir tabla de resultados
    print_results_table(results_df)

    # 10. Guardar modelos
    save_models(trained_models, scaler, results_df)

    print("\n" + "=" * 80)
    print("[OK] PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)

    return trained_models, scaler, results_df


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    models, scaler, results = main()
