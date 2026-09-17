# Definición del Proyecto: Sistema de Triaje Inteligente con Machine Learning

## 1. Nombre del Proyecto

**"Triaje Predictivo"** — Sistema de apoyo a la clasificación de prioridad de pacientes basado en síntomas y signos vitales.

---

## 2. Planteamiento del Problema

En servicios de urgencias, el triaje es el proceso mediante el cual se clasifica a los pacientes según su gravedad para determinar el orden de atención. El principio fundamental es que "lo urgente no siempre es grave y lo grave no siempre es urgente". Este proceso requiere personal capacitado y consume tiempo valioso, especialmente en contextos con recursos limitados.

**La pregunta central**: ¿Puede un modelo de Machine Learning, entrenado con datos de síntomas y signos vitales, predecir automáticamente el nivel de gravedad de un paciente para asistir al personal de triaje?

---

## 3. Dataset Seleccionado

**Fuente**: Proyecto de código abierto alojado en GitHub `bouramah/projet_data_mining_sante`.

**Estructura del dataset**:

| Tipo                                | Variables                                                                    |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| **Demográficas**             | Age (edad), Gender (género)                                                 |
| **Síntomas**                 | Symptom_1, Symptom_2, Symptom_3 (descripciones textuales)                    |
| **Signos Vitales**            | Heart_Rate_bpm, Body_Temperature_C, Blood_Pressure_mmHg, Oxygen_Saturation_% |
| **Etiqueta 1 (Diagnóstico)** | Healthy, Flu, Cold, Bronchitis, Pneumonia                                    |
| **Etiqueta 2 (Gravedad)**     | Mild, Moderate, Severe                                                       |
| **Etiqueta 3 (Tratamiento)**  | Rest and fluids, Medication and rest, Hospitalization and medication         |

**Ventaja clave**: Contiene etiquetas de gravedad (Severity) ya definidas, lo que permite entrenar un modelo de clasificación directo para el triaje.

---

## 4. Enfoque Técnico

### 4.1 Fase de Entrenamiento del Modelo

**Problema**: Clasificación multiclase (predecir una de tres severidades: Mild, Moderate, Severe).

**Preprocesamiento** (siguiendo la estructura del proyecto original):

- Limpieza de duplicados y valores nulos
- Transformación de `Blood_Pressure_mmHg` (formato "120/80") en dos variables: `BP_Systolic` y `BP_Diastolic`
- One-Hot Encoding para variables categóricas (Gender, síntomas)
- Normalización de variables numéricas con MinMaxScaler

**Modelo seleccionado**: **Random Forest (Bosque Aleatorio)**

Justificación basada en evidencia:

- En un estudio reciente sobre clasificación de enfermedades respiratorias, Random Forest alcanzó una precisión del **93%**, superando a Logistic Regression, Decision Tree, KNN, XGBoost y MLP
- En otro estudio con datos de EHR pediátricos, Random Forest obtuvo la precisión más alta en hold-out test (**0.8578**)
- Es robusto, interpretable (importancia de características) y no requiere GPU

**Alternativas a evaluar** (para comparación en la presentación):

- Logistic Regression (baseline)
- XGBoost (candidate)

**Métricas de evaluación**:

- Accuracy, Precision, Recall, F1-Score (por clase)
- Matriz de confusión
- Importancia de características (feature importance)

### 4.2 Arquitectura de la Solución

```
┌─────────────────────────────────────────────────────────────┐
│                    FASE 1: ENTRENAMIENTO                     │
│  Dataset CSV → Preprocesamiento → Random Forest → Modelo     │
│  (.pkl)                                                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    FASE 2: DESPLIEGUE                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              APLICACIÓN STREAMLIT                     │   │
│  │                                                       │   │
│  │  Inputs del Usuario:                                 │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │ • Edad (st.number_input)                    │     │   │
│  │  │ • Género (st.selectbox)                     │     │   │
│  │  │ • Síntomas (st.selectbox múltiple)          │     │   │
│  │  │ • Frecuencia cardíaca (st.number_input)     │     │   │
│  │  │ • Temperatura (st.number_input)             │     │   │
│  │  │ • Presión arterial (st.number_input)        │     │   │
│  │  │ • Saturación O₂ (st.number_input)           │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  │                         ↓                            │   │
│  │              [Botón: "Evaluar Triaje"]               │   │
│  │                         ↓                            │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │         RESULTADO DEL TRIAJE                │     │   │
│  │  │                                             │     │   │
│  │  │  🔴 SEVERO - Atención Inmediata             │     │   │
│  │  │  🟡 MODERADO - Atención en <60 min          │     │   │
│  │  │  🟢 LEVE - Puede esperar                    │     │   │
│  │  │                                             │     │   │
│  │  │  (Mostrar nivel de confianza del modelo)    │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Definición de la Aplicación Streamlit

### 5.1 Propósito de la App

Simular un **sistema de triaje automático** donde el usuario (personal de salud o paciente) ingresa los datos del paciente y el modelo predice el nivel de gravedad, mostrando una recomendación de prioridad.

### 5.2 Componentes de la Interfaz

**Sección 1: Datos Demográficos**

- `st.number_input("Edad", min_value=0, max_value=120, value=30)`
- `st.selectbox("Género", ["Masculino", "Femenino", "Otro"])`

**Sección 2: Síntomas**

- `st.multiselect("Seleccione síntomas", opciones_sintomas)` — permite seleccionar 1-3 síntomas
- Los síntomas se codifican como variables binarias para el modelo

**Sección 3: Signos Vitales**

- `st.number_input("Frecuencia Cardíaca (bpm)", min_value=30, max_value=220, value=80)`
- `st.number_input("Temperatura (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1)`
- `st.number_input("Presión Sistólica (mmHg)", min_value=70, max_value=250, value=120)`
- `st.number_input("Presión Diastólica (mmHg)", min_value=40, max_value=150, value=80)`
- `st.number_input("Saturación O₂ (%)", min_value=70, max_value=100, value=98)`

**Sección 4: Resultado del Triaje**

- Botón: `st.button("Evaluar Prioridad")`
- Al hacer clic: el modelo predice Severity
- Visualización del resultado con código de color:
  - 🔴 **SEVERO** → "Atención inmediata requerida"
  - 🟡 **MODERADO** → "Atención en menos de 60 minutos"
  - 🟢 **LEVE** → "Atención no urgente"
- Mostrar `st.progress()` con la probabilidad de cada clase

**Sección 5: Disclaimer**

- Advertencia: "Esta herramienta es solo para fines educativos y de investigación. No constituye diagnóstico médico."

### 5.3 Flujo de la App

1. Usuario llena el formulario con datos del paciente
2. Presiona "Evaluar Prioridad"
3. App preprocesa los datos (mismas transformaciones del entrenamiento)
4. Carga el modelo `.pkl` con `joblib`
5. Predice la clase y probabilidades
6. Muestra el nivel de triaje con color y recomendación
7. Opcional: muestra las 3 probabilidades (Mild, Moderate, Severe)

---

## 6. Entregables

| Entregable                          | Descripción                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| **Notebook de entrenamiento** | Script`.ipynb` con preprocesamiento, entrenamiento y exportación del modelo |
| **Archivo del modelo**        | `triaje_model.pkl` (Random Forest entrenado)                                 |
| **App Streamlit**             | `app.py` con el formulario de triaje                                         |
| **requirements.txt**          | Dependencias: streamlit, pandas, scikit-learn, joblib, numpy                   |
| **Enlace público**           | URL de Streamlit Community Cloud                                               |
| **Presentación**             | 6 diapositivas (máximo)                                                       |

---

## 7. Justificación del Valor del Proyecto

**¿A quién sirve?**

- Personal de triaje en servicios de urgencias con alta demanda
- Centros de salud rurales con pocos médicos
- Como herramienta de apoyo educativo para estudiantes de medicina

**¿Qué decisión mejora?**

- Acelera la clasificación inicial de pacientes
- Reduce la carga cognitiva del personal de triaje
- Puede implementarse como sistema de pre-triaje digital

**Nota sobre limitaciones** (para la sección de conclusiones):

- El dataset es sintético/simplificado
- No reemplaza el juicio clínico profesional
- Requiere validación con datos reales antes de uso clínico

---

## 8. Alcance Excluido (para esta fase)

- ❌ **Chatbot conversacional**: Se implementará en el examen final con Hugging Face o LLM
- ❌ **Predicción de tratamiento**: El foco es solo Severity (gravedad)
- ❌ **Datos reales de Ecuador**: El dataset es de código abierto (no ecuatoriano)

---

## 9. Cronograma de Implementación

| Fase | Actividad                                        | Tiempo estimado |
| ---- | ------------------------------------------------ | --------------- |
| 1    | Descargar dataset, explorar, preprocesar         | 2-3 horas       |
| 2    | Entrenar y evaluar modelos (RF, LogReg, XGBoost) | 2-3 horas       |
| 3    | Exportar modelo, construir app Streamlit local   | 3-4 horas       |
| 4    | Subir a GitHub, desplegar en Streamlit Cloud     | 1-2 horas       |
| 5    | Preparar 6 diapositivas y practicar              | 2 horas         |

---

## 10. Criterios de Éxito (alineados con la rúbrica)

| Criterio                                  | Peso | Cómo se cumple                                                             |
| ----------------------------------------- | ---- | --------------------------------------------------------------------------- |
| Definición y Justificación del Problema | 15%  | Sección 2 de este documento; triaje es problema real con evidencia         |
| Uso de Herramientas del Módulo           | 30%  | Random Forest, preprocesamiento, clasificación multiclase                  |
| Implementación y Resultados              | 35%  | App desplegada en Streamlit Cloud con enlace público                       |
| Presentación (6 diapositivas)            | 10%  | Estructura: Problema → Justificación → Solución → Demo → Conclusiones |
| Conclusiones y Futuras Mejoras            | 10%  | Incluir limitaciones y trabajo futuro (chatbot, datos reales)               |

---

## Resumen Ejecutivo

**Proyecto**: Sistema de triaje inteligente que predice la gravedad de un paciente (Mild/Moderate/Severe) a partir de síntomas y signos vitales.

**Dataset**: `bouramah/projet_data_mining_sante` (GitHub) — contiene síntomas, signos vitales y etiqueta de Severity.

**Modelo**: Random Forest (evidencia de >90% accuracy en literatura similar).

**App**: Streamlit con formulario de entrada y visualización de triaje con código de color.

**Excluido**: Chatbot (para examen final).

**Despliegue**: Streamlit Community Cloud vía GitHub.
