# Sistema de Predicción de Enfermedades Cardíacas

Sistema de Machine Learning para clasificación de enfermedades cardiovasculares con interfaz web interactiva.

## 🚀 Inicio Rápido

### 1. Instalación
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar Modelos
```bash
python train_models.py
```

### 3. Ejecutar Aplicación Web
```bash
streamlit run app_streamlit.py
```

## 📁 Estructura del Proyecto

```
MLS-clasificacion/
├── app_streamlit.py          # Aplicación web
├── train_models.py           # Script de entrenamiento
├── heart.csv                 # Dataset
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
├── .gitignore               # Archivos a ignorar
├── venv/                    # Entorno virtual
└── outputs/                 # Archivos generados
    ├── model_*.pkl          # Modelos entrenados
    ├── scaler.pkl           # Normalizador
    ├── resultados_*.png     # Gráficos
    └── resultados_metricas.csv
```

## 🎯 Modelos Implementados

| Modelo | Tipo | Accuracy |
|--------|------|----------|
| **Random Forest** | Caja Negra | **100%** |
| Árbol de Decisión | Caja Blanca | 98.54% |
| SVM | Caja Negra | 92.20% |
| Regresión Logística | Caja Blanca | 80.98% |

## 📊 Características del Dataset

- **Registros:** 1,025 pacientes
- **Variables:** 13 características + 1 objetivo
- **Balance:** 51.3% con enfermedad, 48.7% sin enfermedad
- **Sin valores nulos**

## 🔧 Funcionalidades

### Aplicación Web (Streamlit)
- **Exploración:** Análisis visual de datos
- **Entrenamiento:** Configuración y entrenamiento de modelos
- **Resultados:** Comparación de métricas y visualizaciones
- **Predicción:** Predicción en tiempo real para nuevos casos

### Script de Entrenamiento
- Preprocesamiento automático
- Validación cruzada (K=5)
- Generación de gráficos
- Guardado de modelos

## ⚠️ Importante

**Este sistema es solo para fines educativos y de investigación. NO reemplaza el diagnóstico médico profesional.**

## 📚 Tecnologías

- **Python 3.8+**
- **Scikit-learn** - Machine Learning
- **Streamlit** - Interfaz web
- **Pandas** - Manipulación de datos
- **Matplotlib/Seaborn** - Visualizaciones

## 👨‍💻 Autor

Proyecto desarrollado para el curso de Inteligencia Artificial - UNMSM