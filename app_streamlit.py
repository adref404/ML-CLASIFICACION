# app_streamlit.py
# Interfaz Gráfica para el Sistema de Predicción de Enfermedades Cardíacas

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Predicción de Enfermedades Cardíacas",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0068C9;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0068C9;
    }
    .success-box {
        background-color: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #C3E6CB;
    }
    .warning-box {
        background-color: #FFF3CD;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #FFEEBA;
    }
    .danger-box {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #F5C6CB;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def load_data():
    """Carga el dataset de enfermedades cardíacas"""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: No se encontró el archivo 'heart.csv'")
        st.stop()

def preprocess_data(df):
    """Preprocesa los datos"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler

def train_models(X_train, y_train):
    """Entrena todos los modelos"""
    models = {
        'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000),
        'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    trained_models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("✅ Entrenamiento completado")
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evalúa los modelos y retorna métricas"""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results.append({
            'Modelo': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    return results_df

def plot_metrics_comparison(results_df):
    """Genera gráfico de comparación de métricas"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Modelo'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    return fig

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Genera matriz de confusión"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                yticklabels=['Sin Enfermedad', 'Con Enfermedad'],
                cbar_kws={'label': 'Cantidad'})
    ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    
    return fig

# ============================================================================
# INICIALIZACIÓN DE ESTADO
# ============================================================================

if 'models' not in st.session_state:
    st.session_state.models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🫀 Sistema de Predicción de Enfermedades Cardíacas</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("📋 Menú de Navegación")
page = st.sidebar.radio(
    "Seleccione una opción:",
    ["🏠 Inicio", "📊 Exploración de Datos", "🤖 Entrenamiento", "📈 Resultados", "🔮 Predicción Individual"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Información del Proyecto:**
    
    - Dataset: Heart Disease UCI
    - Registros: 303
    - Variables: 14
    - Modelos: 4 (2 blancos, 2 negros)
    - Validación: K-Fold (K=5)
    - División: 80% train / 20% test
    """
)

# ============================================================================
# PÁGINA: INICIO
# ============================================================================

if page == "🏠 Inicio":
    st.markdown('<p class="sub-header">Bienvenido al Sistema de Predicción</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 Acerca del Proyecto
        
        Este sistema utiliza **Machine Learning** para predecir la presencia de enfermedades 
        cardíacas en pacientes basándose en características clínicas y demográficas.
        
        #### 🎯 Objetivo
        Proporcionar una herramienta de apoyo diagnóstico que permita identificar pacientes 
        en riesgo de manera temprana, facilitando la toma de decisiones médicas.
        
        #### 🔬 Metodología
        - **Modelos de Caja Blanca (Interpretables):**
          - Regresión Logística
          - Árbol de Decisión
        
        - **Modelos de Caja Negra (Complejos):**
          - Random Forest
          - Support Vector Machine (SVM)
        
        #### 📊 Métricas de Evaluación
        - Accuracy (Exactitud)
        - Precision (Precisión)
        - Recall (Sensibilidad)
        - F1-Score (Media armónica)
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Inicio Rápido
        
        1. **Exploración:** Visualice los datos
        2. **Entrenamiento:** Entrene los modelos
        3. **Resultados:** Compare métricas
        4. **Predicción:** Pruebe con nuevos casos
        """)
        
        st.markdown('<div class="warning-box">⚠️ <strong>Importante:</strong> Este sistema es una herramienta de apoyo y NO reemplaza el diagnóstico médico profesional.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información de las variables
    st.markdown("### 📋 Variables del Dataset")
    
    variables_info = {
        'Variable': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
        'Descripción': [
            'Edad del paciente',
            'Sexo (1=masculino, 0=femenino)',
            'Tipo de dolor torácico (0-3)',
            'Presión arterial en reposo (mm Hg)',
            'Colesterol sérico (mg/dl)',
            'Glucosa en ayunas > 120 mg/dl',
            'Resultados electrocardiográficos',
            'Frecuencia cardíaca máxima',
            'Angina inducida por ejercicio',
            'Depresión del ST',
            'Pendiente del segmento ST',
            'Número de vasos principales',
            'Talasemia',
            'Presencia de enfermedad (0=no, 1=sí)'
        ],
        'Tipo': ['Numérica', 'Categórica', 'Categórica', 'Numérica', 'Numérica',
                 'Binaria', 'Categórica', 'Numérica', 'Binaria', 'Numérica',
                 'Categórica', 'Numérica', 'Categórica', 'Binaria (Target)']
    }
    
    st.dataframe(pd.DataFrame(variables_info), use_container_width=True)

# ============================================================================
# PÁGINA: EXPLORACIÓN DE DATOS
# ============================================================================

elif page == "📊 Exploración de Datos":
    st.markdown('<p class="sub-header">Exploración del Dataset</p>', unsafe_allow_html=True)
    
    # Cargar datos
    df = load_data()
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", df.shape[0])
    with col2:
        st.metric("Variables", df.shape[1])
    with col3:
        st.metric("Con Enfermedad", df[df['target']==1].shape[0])
    with col4:
        st.metric("Sin Enfermedad", df[df['target']==0].shape[0])
    
    st.markdown("---")
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📊 Estadísticas", "📈 Distribuciones", "🔍 Correlaciones"])
    
    with tab1:
        st.markdown("#### Vista del Dataset")
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Información del Dataset")
            buffer = df.dtypes.to_frame('Tipo de Dato')
            st.dataframe(buffer)
        
        with col2:
            st.markdown("##### Valores Nulos")
            null_counts = df.isnull().sum().to_frame('Cantidad')
            st.dataframe(null_counts)
    
    with tab2:
        st.markdown("#### Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab3:
        st.markdown("#### Distribución de la Variable Objetivo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            target_counts = df['target'].value_counts()
            colors = ['#4ECDC4', '#FF6B6B']
            ax.bar(['Sin Enfermedad', 'Con Enfermedad'], target_counts.values, color=colors)
            ax.set_ylabel('Cantidad', fontweight='bold')
            ax.set_title('Distribución de Clases', fontweight='bold', fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            for i, v in enumerate(target_counts.values):
                ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(target_counts.values, labels=['Sin Enfermedad', 'Con Enfermedad'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Proporción de Clases', fontweight='bold', fontsize=14)
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### Distribución de Variables Numéricas")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('target')
        
        selected_var = st.selectbox("Seleccione una variable:", numeric_cols)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(df[selected_var], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(selected_var, fontweight='bold')
        axes[0].set_ylabel('Frecuencia', fontweight='bold')
        axes[0].set_title(f'Histograma - {selected_var}', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Boxplot por clase
        df.boxplot(column=selected_var, by='target', ax=axes[1], patch_artist=True)
        axes[1].set_xlabel('Clase (0=Sin, 1=Con Enfermedad)', fontweight='bold')
        axes[1].set_ylabel(selected_var, fontweight='bold')
        axes[1].set_title(f'Boxplot - {selected_var} por Clase', fontweight='bold')
        plt.suptitle('')
        
        st.pyplot(fig)
    
    with tab4:
        st.markdown("#### Matriz de Correlación")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax, cbar_kws={'label': 'Correlación'})
        ax.set_title('Matriz de Correlación de Variables', fontweight='bold', fontsize=14)
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### Correlación con Variable Objetivo")
        
        target_corr = correlation_matrix['target'].sort_values(ascending=False)
        target_corr = target_corr.drop('target')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in target_corr.values]
        ax.barh(target_corr.index, target_corr.values, color=colors)
        ax.set_xlabel('Correlación con Target', fontweight='bold')
        ax.set_title('Correlación de Variables con Enfermedad Cardíaca', 
                     fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)

# ============================================================================
# PÁGINA: ENTRENAMIENTO
# ============================================================================

elif page == "🤖 Entrenamiento":
    st.markdown('<p class="sub-header">Entrenamiento de Modelos</p>', unsafe_allow_html=True)
    
    # Cargar datos
    df = load_data()
    
    st.markdown("""
    ### Configuración del Entrenamiento
    
    - **División de datos:** 80% entrenamiento / 20% prueba
    - **Validación cruzada:** K-Fold con K=5
    - **Hiperparámetros:** Valores por defecto
    - **Semilla aleatoria:** 42 (para reproducibilidad)
    """)
    
    st.markdown("---")
    
    # Selección de modelos
    st.markdown("#### Selección de Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Modelos de Caja Blanca (Interpretables)**")
        model1 = st.checkbox("Regresión Logística", value=True)
        model2 = st.checkbox("Árbol de Decisión", value=True)
    
    with col2:
        st.markdown("**🧠 Modelos de Caja Negra (Complejos)**")
        model3 = st.checkbox("Random Forest", value=True)
        model4 = st.checkbox("Support Vector Machine (SVM)", value=True)
    
    st.markdown("---")
    
    # Botón de entrenamiento
    if st.button("🚀 ENTRENAR MODELOS", type="primary", use_container_width=True):
        
        with st.spinner("Preparando datos..."):
            # Preprocesar datos
            X, y, scaler = preprocess_data(df)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
        
        st.success(f"✅ Datos preparados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
        
        # Entrenar modelos
        st.markdown("#### Entrenamiento en Progreso")
        trained_models = train_models(X_train, y_train)
        
        st.session_state.models = trained_models
        
        # Evaluar modelos
        with st.spinner("Evaluando modelos..."):
            results = evaluate_models(trained_models, X_test, y_test)
            st.session_state.results = results
        
        st.markdown('<div class="success-box">✅ <strong>Entrenamiento completado exitosamente</strong><br>Los modelos están listos para ser evaluados.</div>', unsafe_allow_html=True)
        
        # Mostrar preview de resultados
        st.markdown("#### Preview de Resultados")
        st.dataframe(results.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}'
        }).background_gradient(cmap='YlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
        use_container_width=True)
        
        st.info("💡 Dirígete a la sección **'Resultados'** para ver el análisis completo.")
    
    # Información adicional
    if st.session_state.models is None:
        st.markdown("---")
        st.markdown("""
        ### ℹ️ Información sobre los Modelos
        
        **Regresión Logística:**
        - Modelo lineal simple e interpretable
        - Estima probabilidades usando función sigmoide
        - Rápido de entrenar y evaluar
        
        **Árbol de Decisión:**
        - Estructura jerárquica de decisiones
        - Altamente interpretable (visualización clara)
        - Puede capturar relaciones no lineales
        
        **Random Forest:**
        - Ensemble de múltiples árboles
        - Mayor precisión y robustez
        - Reduce sobreajuste
        
        **Support Vector Machine (SVM):**
        - Encuentra hiperplano óptimo de separación
        - Efectivo en alta dimensionalidad
        - Usa kernel RBF para no linealidad
        """)

# ============================================================================
# PÁGINA: RESULTADOS
# ============================================================================

elif page == "📈 Resultados":
    st.markdown('<p class="sub-header">Resultados y Métricas</p>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("⚠️ Primero debe entrenar los modelos en la sección **'Entrenamiento'**")
        st.stop()
    
    results_df = st.session_state.results
    
    # Mejor modelo
    best_model = results_df.iloc[0]
    
    st.markdown('<div class="success-box">🏆 <strong>MEJOR MODELO: ' + best_model['Modelo'] + 
                f"</strong><br>Accuracy: {best_model['Accuracy']:.2%} | " +
                f"F1-Score: {best_model['F1-Score']:.2%}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs para resultados
    tab1, tab2, tab3 = st.tabs(["📊 Tabla de Métricas", "📈 Gráficos Comparativos", "🔍 Matrices de Confusión"])
    
    with tab1:
        st.markdown("#### Tabla Comparativa de Métricas")
        
        # Formatear tabla
        styled_df = results_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}'
        }).background_gradient(cmap='YlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("---")
        
        # Métricas individuales
        st.markdown("#### Desglose por Modelo")
        
        for idx, row in results_df.iterrows():
            with st.expander(f"📊 {row['Modelo']}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{row['Accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{row['Precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{row['Recall']:.2%}")
                with col4:
                    st.metric("F1-Score", f"{row['F1-Score']:.2%}")
    
    with tab2:
        st.markdown("#### Gráficos Comparativos")
        
        # Gráfico de barras
        fig = plot_metrics_comparison(results_df)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Gráfico de radar
        st.markdown("#### Gráfico de Radar - Comparación Multidimensional")
        
        from math import pi
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        N = len(categories)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, row in results_df.iterrows():
            values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Modelo'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0.7, 1.0)
        ax.set_title('Comparación Multidimensional de Modelos', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        st.pyplot(fig)
    
    with tab3:
        st.markdown("#### Matrices de Confusión")
        
        selected_model_name = st.selectbox(
            "Seleccione un modelo:",
            results_df['Modelo'].tolist()
        )
        
        model = st.session_state.models[selected_model_name]
        fig = plot_confusion_matrix(model, st.session_state.X_test, 
                                    st.session_state.y_test, selected_model_name)
        st.pyplot(fig)
        
        # Interpretación
        y_pred = model.predict(st.session_state.X_test)
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Interpretación")
            st.markdown(f"""
            - **Verdaderos Negativos (TN):** {cm[0,0]} - Correctamente identificados como sanos
            - **Falsos Positivos (FP):** {cm[0,1]} - Incorrectamente identificados como enfermos
            - **Falsos Negativos (FN):** {cm[1,0]} - Incorrectamente identificados como sanos
            - **Verdaderos Positivos (TP):** {cm[1,1]} - Correctamente identificados como enfermos
            """)
        
        with col2:
            st.markdown("##### Métricas Derivadas")
            sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
            specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
            
            st.metric("Sensibilidad (Recall)", f"{sensitivity:.2%}")
            st.metric("Especificidad", f"{specificity:.2%}")
    
    st.markdown("---")
    
    # Conclusiones
    st.markdown("### 📝 Conclusiones")
    
    st.markdown(f"""
    **Análisis de Resultados:**
    
    1. **Mejor Desempeño:** {best_model['Modelo']} con un accuracy de {best_model['Accuracy']:.2%}
    
    2. **Balance Precision-Recall:** Un F1-Score de {best_model['F1-Score']:.2%} indica un buen equilibrio 
       entre la identificación de casos positivos (enfermos) y la minimización de falsos positivos.
    
    3. **Aplicabilidad Clínica:** 
       - Alta sensibilidad minimiza el riesgo de no detectar pacientes enfermos (crítico en medicina)
       - Precisión adecuada evita alarmas innecesarias y pruebas invasivas sin necesidad
    
    4. **Recomendación:** El modelo {best_model['Modelo']} es recomendado para implementación como 
       herramienta de screening inicial, siempre bajo supervisión médica profesional.
    """)

# ============================================================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ============================================================================

elif page == "🔮 Predicción Individual":
    st.markdown('<p class="sub-header">Predicción para Nuevo Paciente</p>', unsafe_allow_html=True)
    
    if st.session_state.models is None:
        st.warning("⚠️ Primero debe entrenar los modelos en la sección **'Entrenamiento'**")
        st.stop()
    
    st.markdown("""
    ### Ingrese los datos del paciente
    Complete el formulario con la información clínica del paciente para obtener una predicción.
    """)
    
    st.markdown("---")
    
    # Formulario de entrada
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Datos Demográficos")
            age = st.number_input("Edad", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
            
            st.markdown("##### 💊 Indicadores Químicos")
            chol = st.number_input("Colesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Glucosa en ayunas > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        
        with col2:
            st.markdown("##### 🫀 Síntomas y Dolor")
            cp = st.selectbox("Tipo de dolor torácico", options=[0, 1, 2, 3], 
                            format_func=lambda x: {0: "Típico anginal", 1: "Atípico anginal", 
                                                  2: "No anginal", 3: "Asintomático"}[x])
            exang = st.selectbox("Angina inducida por ejercicio", options=[0, 1], 
                               format_func=lambda x: "No" if x == 0 else "Sí")
            
            st.markdown("##### 🏃 Ejercicio")
            thalach = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
            oldpeak = st.number_input("Depresión del ST", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        with col3:
            st.markdown("##### 🩺 Mediciones Clínicas")
            trestbps = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=200, value=120)
            restecg = st.selectbox("Resultados ECG", options=[0, 1, 2],
                                 format_func=lambda x: {0: "Normal", 1: "Anormalidad ST-T", 2: "Hipertrofia ventricular"}[x])
            
            st.markdown("##### 🔬 Pruebas Especializadas")
            slope = st.selectbox("Pendiente segmento ST", options=[0, 1, 2],
                               format_func=lambda x: {0: "Ascendente", 1: "Plano", 2: "Descendente"}[x])
            ca = st.number_input("Número de vasos principales", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Talasemia", options=[1, 2, 3],
                              format_func=lambda x: {1: "Normal", 2: "Defecto fijo", 3: "Defecto reversible"}[x])
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("🔮 REALIZAR PREDICCIÓN", use_container_width=True, type="primary")
    
    # Realizar predicción
    if submit_button:
        # Crear DataFrame con los datos ingresados
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        # Normalizar datos
        scaler = st.session_state.scaler
        input_scaled = scaler.transform(input_data)
        
        st.markdown("---")
        st.markdown("### 📊 Resultados de la Predicción")
        
        # Predicción con todos los modelos
        predictions = {}
        probabilities = {}
        
        for name, model in st.session_state.models.items():
            pred = model.predict(input_scaled)[0]
            predictions[name] = pred
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_scaled)[0]
                probabilities[name] = prob[1]  # Probabilidad de clase positiva
            else:
                probabilities[name] = None
        
        # Usar el mejor modelo para la predicción principal
        best_model_name = st.session_state.results.iloc[0]['Modelo']
        best_prediction = predictions[best_model_name]
        best_probability = probabilities[best_model_name]
        
        # Mostrar resultado principal
        if best_prediction == 1:
            st.markdown(f'<div class="danger-box">⚠️ <strong>ALERTA: RIESGO DE ENFERMEDAD CARDÍACA DETECTADO</strong><br><br>'
                       f'El modelo <strong>{best_model_name}</strong> ha identificado indicadores de posible enfermedad cardíaca.<br>'
                       f'Probabilidad estimada: <strong>{best_probability:.1%}</strong></div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ <strong>RESULTADO: BAJO RIESGO DE ENFERMEDAD CARDÍACA</strong><br><br>'
                       f'El modelo <strong>{best_model_name}</strong> no ha detectado indicadores significativos de enfermedad cardíaca.<br>'
                       f'Probabilidad de enfermedad: <strong>{best_probability:.1%}</strong></div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Consenso de todos los modelos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 Predicciones por Modelo")
            
            pred_df = pd.DataFrame({
                'Modelo': list(predictions.keys()),
                'Predicción': [('🔴 Con Enfermedad' if p == 1 else '🟢 Sin Enfermedad') for p in predictions.values()],
                'Probabilidad': [f"{p:.2%}" if p is not None else "N/A" for p in probabilities.values()]
            })
            
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # Consenso
            positive_count = sum(predictions.values())
            consensus = positive_count / len(predictions)
            
            st.metric("Consenso de Modelos", f"{positive_count}/{len(predictions)} modelos detectan enfermedad")
            
            if consensus >= 0.75:
                st.error("⚠️ Alto consenso en detección de enfermedad")
            elif consensus >= 0.5:
                st.warning("⚡ Consenso moderado - Se recomienda evaluación adicional")
            else:
                st.success("✅ Consenso bajo en detección de enfermedad")
        
        with col2:
            st.markdown("#### 📊 Distribución de Probabilidades")
            
            # Gráfico de probabilidades
            prob_data = {k: v for k, v in probabilities.items() if v is not None}
            
            if prob_data:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#FF6B6B' if predictions[k] == 1 else '#4ECDC4' for k in prob_data.keys()]
                bars = ax.barh(list(prob_data.keys()), list(prob_data.values()), color=colors)
                
                ax.set_xlabel('Probabilidad de Enfermedad', fontweight='bold')
                ax.set_xlim([0, 1])
                ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_title('Probabilidades por Modelo', fontweight='bold', fontsize=12)
                ax.grid(axis='x', alpha=0.3)
                
                # Añadir valores en las barras
                for i, (bar, prob) in enumerate(zip(bars, prob_data.values())):
                    ax.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontweight='bold')
                
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Recomendaciones
        st.markdown("### 💡 Recomendaciones")
        
        if best_prediction == 1:
            st.markdown("""
            **🏥 Acciones Recomendadas:**
            
            1. **Consulta Médica Urgente:** Agende una cita con un cardiólogo lo antes posible
            2. **Estudios Adicionales:** Es probable que requiera:
               - Electrocardiograma (ECG)
               - Ecocardiograma
               - Prueba de esfuerzo
               - Análisis de sangre completo
            3. **Monitoreo:** Registre cualquier síntoma como dolor torácico, fatiga o dificultad respiratoria
            4. **Estilo de Vida:** Considere modificaciones inmediatas:
               - Reducir actividad física intensa
               - Evitar situaciones de estrés
               - Dieta baja en sodio y grasas saturadas
            
            ⚠️ **IMPORTANTE:** Esta predicción es una herramienta de apoyo. Solo un médico puede dar un diagnóstico definitivo.
            """)
        else:
            st.markdown("""
            **✅ Recomendaciones Preventivas:**
            
            1. **Mantener Hábitos Saludables:**
               - Dieta balanceada rica en frutas, vegetales y granos integrales
               - Ejercicio regular (150 minutos semanales de actividad moderada)
               - Control de peso corporal
            
            2. **Chequeos Regulares:**
               - Monitoreo anual de presión arterial
               - Control de colesterol cada 2-3 años
               - Evaluación médica general anual
            
            3. **Factores de Riesgo:**
               - No fumar
               - Limitar consumo de alcohol
               - Gestionar el estrés
               - Dormir 7-8 horas diarias
            
            4. **Vigilancia de Síntomas:**
               - Estar atento a cambios en salud cardiovascular
               - Consultar médico ante cualquier síntoma inusual
            
            ℹ️ **NOTA:** Aunque el riesgo actual es bajo, la prevención continua es fundamental.
            """)
        
        st.markdown("---")
        
        # Resumen de datos ingresados
        with st.expander("📋 Ver Resumen de Datos Ingresados"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Información del Paciente:**")
                st.write(f"- Edad: {age} años")
                st.write(f"- Sexo: {'Masculino' if sex == 1 else 'Femenino'}")
                st.write(f"- Presión arterial: {trestbps} mm Hg")
                st.write(f"- Colesterol: {chol} mg/dl")
                st.write(f"- Frecuencia cardíaca máxima: {thalach} bpm")
                st.write(f"- Glucosa en ayunas > 120: {'Sí' if fbs == 1 else 'No'}")
            
            with col2:
                st.markdown("**Datos Clínicos:**")
                st.write(f"- Tipo de dolor torácico: {['Típico', 'Atípico', 'No anginal', 'Asintomático'][cp]}")
                st.write(f"- Angina por ejercicio: {'Sí' if exang == 1 else 'No'}")
                st.write(f"- Depresión ST: {oldpeak}")
                st.write(f"- Pendiente ST: {['Ascendente', 'Plano', 'Descendente'][slope]}")
                st.write(f"- Vasos principales: {ca}")
                st.write(f"- Talasemia: {['', 'Normal', 'Defecto fijo', 'Defecto reversible'][thal]}")
        
        # Disclaimer
        st.markdown("---")
        st.markdown('<div class="warning-box">⚠️ <strong>DESCARGO DE RESPONSABILIDAD</strong><br><br>'
                   'Este sistema es una herramienta de apoyo diagnóstico basada en machine learning y NO reemplaza '
                   'el criterio médico profesional. Las predicciones deben ser validadas por un especialista en cardiología. '
                   'Ante cualquier síntoma o duda sobre su salud, consulte con un médico inmediatamente.</div>', 
                   unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>🫀 Sistema de Predicción de Enfermedades Cardíacas</strong></p>
    <p>Desarrollado con Python, Scikit-learn y Streamlit</p>
    <p>Dataset: Heart Disease UCI | Kaggle</p>
    <p style='font-size: 0.8rem;'>© 2025 - Proyecto Académico de Machine Learning</p>
</div>
""", unsafe_allow_html=True)