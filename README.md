# 🏥 Triaje Predictivo: Sistema Inteligente de Clasificación de Pacientes en Urgencias

Aplicación web interactiva desarrollada con **Streamlit** y **Scikit-Learn** para clasificar la gravedad y prioridad de atención de pacientes en servicios de urgencias a partir de constantes vitales, sintomatología y perfiles hemodinámicos.

---

## 🎯 Objetivo del Proyecto

En las salas de urgencias hospitalarias, el triaje estratifica a los pacientes para determinar el orden de atención asistencial. Este proyecto implementa un sistema de apoyo a la decisión clínica (CDSS) entrenado con datos de admisiones hospitalarias, libre de fuga de datos (*data leakage*), que clasifica al paciente en tres niveles de gravedad estandarizados:

- 🔴 **Nivel I — Prioridad Alta (Severo):** Atención médica inmediata en box de reanimación / soporte vital.
- 🟡 **Nivel II — Prioridad Media (Moderado):** Atención urgente en menos de 60 minutos (cuadros febriles agudos o afectación sistémica).
- 🟢 **Nivel III — Prioridad Estándar (Leve):** Atención ambulatoria diferida / sala de espera general.

---

## 📊 Arquitectura y Métricas del Modelo

- **Modelo Seleccionado:** `Random Forest Classifier` (200 estimadores, profundidad máxima 9, balanceo ponderado de clases).
- **Validación Cruzada (5-Fold Stratified CV):**
  - **Macro F1-Score:** 99.87%
  - **Accuracy:** 99.94%
- **Evaluación en Conjunto de Prueba Independiente ($N = 400$, 20% Hold-Out Test):**
  - **Sensibilidad en Casos Severos (Recall Severe):** **100.0%** (cero falsos negativos críticos)
  - **Accuracy Global:** **100.0%**
  - **Macro F1:** **100.0%**
- **Variables con Mayor Importancia Clínica:**
  1. Presencia de fiebre (`sym_fever`): 20.3%
  2. Presencia de tos (`sym_cough`): 17.4%
  3. Temperatura corporal continua (`Body_Temperature_C`): 12.7%
  4. Alerta de hipoxemia (`flag_hypoxia`, $\text{SatO}_2 < 95\%$): 11.1%
  5. Saturación periférica de oxígeno (`Oxygen_Saturation_%`): 10.8%
  6. Alertas hemodinámicas derivadas (`shock_index`, `map_pressure`, `flag_tachycardia`).

---

## 📁 Estructura del Repositorio

```text
13-streamlit/
├── app.py                                   # Aplicación web interactiva en Streamlit
├── requirements.txt                         # Dependencias para despliegue en la nube
├── README.md                                # Documentación de uso y despliegue
├── informe_hallazgos_eda_y_modelos.md       # Informe técnico de bioestadística y ML
├── idea.md                                  # Especificación y fundamentación del proyecto
├── data/
│   └── disease_diagnosis.csv                # Dataset clínico fuente (2000 admisiones)
├── models/
│   ├── triaje_model.joblib                  # Pipeline completo serializado (Extractor + RF)
│   └── model_metadata.json                  # Metadatos, matrices de confusión y métricas
├── notebooks/
│   └── 01_eda_limpieza_modelado.ipynb       # Notebook reproducible paso a paso
├── output/
│   └── eda_summary.json                     # Resumen cuantitativo de pruebas estadísticas
└── src/
    ├── __init__.py
    ├── data_preprocessing.py                # Extractor de features e indicadores médicos
    ├── eda_analysis.py                      # Pruebas de Chi2, V de Cramér y Kruskal-Wallis
    └── train_and_evaluate.py                # Benchmark de 6 modelos y exportación
```

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio y acceder a la carpeta:
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd 13-streamlit
```

### 2. Crear y activar entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate      # En Linux / macOS
# .venv\Scripts\activate       # En Windows
```

### 3. Instalar requerimientos:
```bash
pip install -r requirements.txt
```

### 4. (Opcional) Reentrenar el modelo o ejecutar EDA:
```bash
python src/eda_analysis.py
python src/train_and_evaluate.py
```

### 5. Lanzar la aplicación web:
```bash
streamlit run app.py
```
La interfaz se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## ☁️ Despliegue en Streamlit Community Cloud (Guía Paso a Paso)

1. **Crear repositorio en GitHub:**
   - Sube los archivos de `13-streamlit` a un nuevo repositorio público en tu cuenta de GitHub (ejemplo: `triaje-predictivo-streamlit`).
2. **Iniciar sesión en Streamlit Community Cloud:**
   - Ingresa a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
3. **Crear la aplicación ("New app"):**
   - Haz clic en **"Create app"** o **"New app"**.
   - Selecciona **"I already have an app"** o escoge tu repositorio de GitHub.
   - Configuración:
     - **Repository:** `tu-usuario/triaje-predictivo-streamlit`
     - **Branch:** `main` (o `master`)
     - **Main file path:** `app.py`
     - **App URL:** Elige un subdominio personalizado (ej: `triaje-predictivo.streamlit.app`).
4. **Desplegar:**
   - Haz clic en **"Deploy!"**.
   - En 1-2 minutos la aplicación estará pública y en línea.
5. **Entrega en Moodle:**
   - Copia la URL pública generada (ej: `https://triaje-predictivo.streamlit.app`) y pégala en la entrega de la tarea en la plataforma.

---

## ⚖️ Descargo de Responsabilidad Médico

Este software ha sido diseñado con fines académicos y de investigación en el marco del módulo de Fundamentos de Inteligencia Artificial. No constituye un dispositivo médico certificado ni sustituye la valoración clínica presencial de profesionales de la salud.
