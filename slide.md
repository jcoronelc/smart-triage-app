
# Sistema de Triaje Inteligente con Machine Learning y Graph RAG — Documento Final del Proyecto

**Proyecto:** Triaje Predictivo — Sistema de Apoyo a la Decisión Clínica en Triaje Hospitalario
**Módulo:** Fundamentos de Inteligencia Artificial
**Maestría en Inteligencia Artificial**
**Fecha:** Septiembre 2026


Link: [smart-triage-bot-app-ej2iwgv6nbmf8cfwqsgmce.streamlit.app
](https://smart-triage-bot-app-ej2iwgv6nbmf8cfwqsgmce.streamlit.app)

---

## 1. Introducción al Problema

El **triaje hospitalario** es el proceso mediante el cual se clasifica a los pacientes según su gravedad para determinar el orden de atención en servicios de urgencias. Su propósito no es emitir un diagnóstico etiológico definitivo, sino **estratificar con prontitud la prioridad de atención médica**, bajo el principio rector de que los recursos críticos deben asignarse primero a pacientes con riesgo inminente de colapso hemodinámico o insuficiencia ventilatoria.

El principio fundamental que rige este proceso es que **"lo urgente no siempre es grave y lo grave no siempre es urgente"**. Este proceso requiere personal capacitado y consume tiempo valioso, especialmente en contextos con recursos limitados, como zonas rurales o servicios de urgencias con alta demanda.

**La pregunta central del proyecto es:**

> ¿Puede un modelo de Machine Learning, entrenado con datos de síntomas y signos vitales, predecir automáticamente el nivel de gravedad de un paciente para asistir al personal de triaje?

El problema se aborda desde una perspectiva de **clasificación multiclase**, donde el modelo debe predecir una de tres severidades: **Mild (Leve)**, **Moderate (Moderado)** o **Severe (Severo)**.

---

## 2. Justificación de la Importancia del Problema

### 2.1 Relevancia Clínica

En servicios de urgencias, la **infraclasificación de pacientes críticos** (falsos negativos en la clase `Severe`) conlleva consecuencias clínicas potencialmente fatales. Un sistema de apoyo al triaje que garantice alta sensibilidad en la detección de casos severos puede **reducir el riesgo de mortalidad evitable** y optimizar la asignación de recursos escasos.

### 2.2 Relevancia Operativa

- **Acelera la clasificación inicial de pacientes**, reduciendo el tiempo de espera para casos críticos.
- **Reduce la carga cognitiva del personal de triaje**, que debe procesar múltiples variables simultáneamente bajo presión.
- **Puede implementarse como sistema de pre-triaje digital**, especialmente útil en centros de salud rurales con pocos médicos.

### 2.3 Relevancia Técnica

El proyecto integra técnicas vistas en el módulo:

- **Machine Learning supervisado**: Random Forest, XGBoost, SVM, MLP, Logistic Regression, KNN.
- **Preprocesamiento de datos**: limpieza, ingeniería de características, balanceo, normalización.
- **Clasificación multiclase**: evaluación con métricas insensibles al desbalance.
- **Graph RAG**: recuperación de conocimiento estructurado para anclar respuestas de un LLM.
- **Despliegue**: Streamlit Community Cloud.

### 2.4 Evidencia Científica

Un estudio del *Journal of the American Medical Informatics Association* (JAMIA) demostró que los sistemas de triaje aumentados con conocimiento estructurado externo alcanzan niveles de **sensibilidad del 98%** y **especificidad del 99%**, superando sustancialmente a los modelos de lenguaje aislados. Esto respalda la arquitectura híbrida (ML + Graph RAG + LLM) adoptada en este proyecto.

---

## 3. Descripción Técnica de la Solución

### 3.1 Dataset

**Fuente:** Proyecto de código abierto `bouramah/projet_data_mining_sante` (GitHub).
**Volumen:** 2,000 admisiones hospitalarias.

| Tipo                                | Variables                                                                    |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| **Demográficas**             | Age, Gender                                                                  |
| **Síntomas**                 | Symptom_1, Symptom_2, Symptom_3 (descripciones textuales)                    |
| **Signos Vitales**            | Heart_Rate_bpm, Body_Temperature_C, Blood_Pressure_mmHg, Oxygen_Saturation_% |
| **Etiqueta 1 (Diagnóstico)** | Healthy, Flu, Cold, Bronchitis, Pneumonia                                    |
| **Etiqueta 2 (Gravedad)**     | Mild, Moderate, Severe                                                       |
| **Etiqueta 3 (Tratamiento)**  | Rest and fluids, Medication and rest, Hospitalization and medication         |

**Distribución de la variable objetivo (`Severity`):**

| Nivel              | Conteo (n)      | Porcentaje        | Prioridad Clínica                  |
| ------------------ | --------------- | ----------------- | ----------------------------------- |
| **Mild**     | 1,330           | 66.50%            | Atención diferida / Sala de espera |
| **Severe**   | 378             | 18.90%            | Atención médica inmediata         |
| **Moderate** | 292             | 14.60%            | Atención prioritaria (< 60 min)    |
| **Total**    | **2,000** | **100.00%** | —                                  |

> **Principio de No-Fuga de Datos:** Las variables `Diagnosis` y `Treatment_Plan` representan desenlaces clínicos posteriores, por lo que **quedan estrictamente excluidas** de las variables predictoras del modelo.

---

### 3.2 Análisis Exploratorio y Bioestadístico

#### 3.2.1 Pruebas de Asociación (Chi-cuadrado y V de Cramér)

| Síntoma                      | χ² (g.l.=2) | Valor p        | V de Cramér     | Prevalencia Severe | Interpretación                        |
| ----------------------------- | ------------- | -------------- | ---------------- | ------------------ | -------------------------------------- |
| **Cough (Tos)**         | 556.48        | < 10⁻¹²⁰   | **0.5267** | 91.8%              | Marcador casi universal en severos     |
| **Fever (Fiebre)**      | 548.70        | < 10⁻¹¹⁹   | **0.5230** | 32.3%              | Presente en 100% de moderados          |
| **Shortness of breath** | 47.98         | 3.82×10⁻¹¹ | 0.1517           | 28.8%              | Disnea aislada más frecuente en leves |
| **Fatigue**             | 40.24         | 1.83×10⁻⁹   | 0.1383           | 27.5%              | Inespecífico, predominio leve         |
| **Body ache**           | 38.85         | 3.67×10⁻⁹   | 0.1358           | 29.6%              | Común en cuadros virales leves        |
| **Sore throat**         | 23.25         | 8.93×10⁻⁶   | 0.1031           | 28.3%              | Afección de vía aérea superior      |
| **Runny nose**          | 20.89         | 2.91×10⁻⁵   | 0.0972           | 29.6%              | Típico de rinitis/resfriado           |
| **Headache**            | 13.55         | 0.00114        | 0.0760           | 31.7%              | Débil capacidad discriminante         |
| **Gender**              | 0.28          | 0.8687         | 0.0119           | 50.8% Masc.        | Sin significación                     |

**Hallazgo clave:** La **tos persistente** (V = 0.527) se presenta en el **91.8% de los pacientes severos**, mientras que la **fiebre** (V = 0.523) aparece en el **100% de los moderados**.

#### 3.2.2 Comparación de Signos Vitales (Kruskal-Wallis)

| Constante Vital                | Media Global ± DE | Mediana Leve | Mediana Moderado | Mediana Severo  | H      | Valor p    |
| ------------------------------ | ------------------ | ------------ | ---------------- | --------------- | ------ | ---------- |
| **Saturación O₂ (%)**  | 94.49 ± 2.86      | 95.0%        | 95.0%            | **92.0%** | 371.96 | < 10⁻⁸⁰ |
| **Temperatura (°C)**    | 37.74 ± 1.31      | 37.4         | **39.0**   | 37.7            | 325.02 | < 10⁻⁷⁰ |
| **Frecuencia Cardíaca** | 89.44 ± 17.52     | 90.0         | 85.0             | **92.0**  | 18.36  | 0.000103   |
| **Presión Sistólica**  | 139.75 ± 23.36    | 140.0        | 139.0            | 139.5           | 0.07   | 0.9634     |
| **Presión Diastólica** | 89.34 ± 14.86     | 89.0         | 89.0             | 88.0            | 0.52   | 0.7706     |
| **Edad**                 | 48.29 ± 17.42     | 49.0         | 49.0             | 48.0            | 1.38   | 0.5023     |

#### 3.2.3 Auditoría de Outliers

**Análisis univariado:** 0 outliers (0.0%) en todas las variables continuas. La curtosis en exceso ≈ **-1.20** coincide con la de una distribución uniforme U(a,b), lo que demuestra que los signos vitales fueron generados mediante **muestreo uniforme sintético** dentro de rangos acotados.

**Anomalías multivariadas identificadas:**

1. **Inversión tensional (PAD ≥ PAS):** 156 pacientes (7.8%) — fisiológicamente incompatible con la vida.
2. **Presión de pulso estrecha (< 20 mmHg):** 432 pacientes.
3. **Anomalías con Isolation Forest (3% contaminación):** 60 casos de descompensación extrema (taquicardia ≥ 115 lpm + SatO₂ ≤ 91% + Shock Index > 1.0).

---

### 3.3 Ingeniería de Características Clínicas

Se crearon **22 variables enriquecidas** para capturar la fisiopatología del paciente:

1. **Descomposición tensional:** `BP_Systolic`, `BP_Diastolic`
2. **Binarización de síntomas:** 8 banderas booleanas (`sym_cough`, `sym_fever`, etc.) + `symptom_count`
3. **Índice de Shock:** `Shock Index = Heart_Rate / BP_Systolic` (valores > 0.9 alertan inestabilidad hemodinámica)
4. **Presión Arterial Media (PAM):** `PAM = PAD + (PAS - PAD)/3`
5. **Banderas de riesgo clínico:**
   - `flag_hypoxia`: 1 si SatO₂ < 95%
   - `flag_fever`: 1 si Temp ≥ 38.0°C
   - `flag_tachycardia`: 1 si FC > 100 lpm
   - `flag_elderly`: 1 si Edad ≥ 65 años

> **Nota metodológica:** Las características derivadas explican más del **27% de la capacidad de decisión del modelo**, demostrando el valor de transformar variables crudas en indicadores biomédicos.

---

### 3.4 Benchmark de Modelos

Se evaluaron **6 familias de algoritmos** mediante validación cruzada estratificada de 5 pliegues (80% entrenamiento, n=1600):

| Algoritmo                              | Macro F1         | Accuracy | Balanced Accuracy |
| -------------------------------------- | ---------------- | -------- | ----------------- |
| **XGBoost**                      | **100.0%** | 100.0%   | 100.0%            |
| **Random Forest (Seleccionado)** | **99.87%** | 99.94%   | 99.89%            |
| **SVM (RBF)**                    | 98.79%           | 99.19%   | 99.28%            |
| **MLP (Red Neuronal)**           | 97.97%           | 98.69%   | 98.17%            |
| **Logistic Regression**          | 97.96%           | 98.75%   | 98.60%            |
| **KNN (k=7)**                    | 89.76%           | 92.69%   | 87.16%            |

**Justificación de la elección de Random Forest:**

- **Calibración de probabilidades:** `predict_proba` confiable para cuantificar incertidumbre diagnóstica.
- **Interpretabilidad:** Permite desglosar qué variables motivaron la recomendación clínica.
- **Robustez operativa:** Inferencia sub-milisegunda en CPU sin dependencias pesadas en Streamlit Cloud.

---

### 3.5 Resultados en Conjunto de Prueba (N=400)

```
                  Precision    Recall  F1-Score   Support
        Mild       1.00      1.00      1.00       266
    Moderate       1.00      1.00      1.00        58
      Severe       1.00      1.00      1.00        76

    Accuracy                           1.00       400
   Macro avg       1.00      1.00      1.00       400
Weighted avg       1.00      1.00      1.00       400
```

- **Sensibilidad en casos severos (Recall Severe):** **100.0%** (76/76 detectados, cero falsos negativos)
- **Especificidad:** **100.0%**

---

### 3.6 Importancia de Características (Top 12)

| Ranking | Variable                            | Importancia (Gini) |
| ------- | ----------------------------------- | ------------------ |
| 1       | `sym_fever` (Fiebre sintomática) | 20.33%             |
| 2       | `sym_cough` (Tos persistente)     | 17.42%             |
| 3       | `Body_Temperature_C`              | 12.69%             |
| 4       | `flag_hypoxia`                    | 11.09%             |
| 5       | `Oxygen_Saturation_%`             | 10.82%             |
| 6       | `flag_fever`                      | 10.25%             |
| 7       | `Heart_Rate_bpm`                  | 2.99%              |
| 8       | `flag_tachycardia`                | 2.10%              |
| 9       | `shock_index`                     | 1.89%              |
| 10      | `map_pressure`                    | 1.66%              |

---

## 4. Arquitectura y Flujo del Sistema

### 4.1 Stack Tecnológico

| Capa                              | Tecnología                                   | Función                                              |
| --------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| **Frontend**                | Streamlit                                     | Interfaz web interactiva, formularios, visualización |
| **ML**                      | Scikit-learn (Random Forest, 200 estimadores) | Clasificación determinista de severidad              |
| **Grafo**                   | NetworkX (347 nodos, 575 aristas)             | Fuente única de verdad clínica                      |
| **Persistencia del grafo**  | GraphML, JSON, Pickle                         | Exportación y carga < 2 ms                           |
| **LLM (opcional)**          | Ollama (qwen3.5:9b, deepseek-r1:8b)           | Traducción de hechos a lenguaje natural              |
| **API LLM (alternativa)**   | Groq (Llama 3, Mixtral)                       | Inferencia remota de alta velocidad                   |
| **Despliegue**              | Streamlit Community Cloud                     | Hosting gratuito vía GitHub                          |
| **Control de versiones**    | Git + GitHub                                  | Repositorio, CI/CD                                    |
| **Visualización de grafo** | Vis.js / Gephi / Cytoscape                    | Exploración interactiva de la ontología             |

---

### 4.2 Flujo End-to-End del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1: ADMISIÓN E INFERENCIA DE TRIAJE                            │
│                                                                     │
│  INPUT:  Edad, Sexo, Signos Vitales (FC, PAS, PAD, SatO₂, Temp)    │
│          + Síntomas cardinales (hasta 3)                            │
│                          ↓                                          │
│  PIPELINE ML:  Imputación + Feature Engineering                     │
│                (Shock Index, PAM, Banderas Clínicas)                │
│                          ↓                                          │
│  CLASIFICADOR: Random Forest (200 estimadores)                      │
│                          ↓                                          │
│  OUTPUT: Severidad ('Mild', 'Moderate', 'Severe') + Certeza (%)     │
│                          ↓                                          │
│  PERSISTENCIA: st.session_state.ultimo_paciente_evaluado            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 2: RECUPERACIÓN DE EVIDENCIA ONTOLÓGICA (Graph RAG)           │
│                                                                     │
│  INPUT:  Severidad predicha + Síntomas ingresados                   │
│                          ↓                                          │
│  CONSULTA AL GRAFO: retrieve_clinical_context(severity, symptoms)   │
│                          ↓                                          │
│  EXTRACCIÓN DE NODOS VECINOS:                                       │
│    • REQUIRES_PRECAUTION → Nodos Precaution                         │
│    • RECOMMENDS_MEDICATION → Nodos Medication                       │
│    • SUGGESTS_DIET → Nodos Diet                                     │
│    • HAS_TRIAGE_SEVERITY → Nodos Severity + clinical_action         │
│                          ↓                                          │
│  OUTPUT: graph_ctx (diccionario estructurado con hechos verificados)│
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 3: CONSTRUCCIÓN DE PROMPT BIOÉTICO Y ANCLAJE                  │
│                                                                     │
│  INPUT:  graph_ctx + datos del paciente + severidad ML              │
│                          ↓                                          │
│  COMPILADOR: construir_prompt_sistema_graph_rag()                   │
│                          ↓                                          │
│  OUTPUT: System Prompt Clínico con:                                 │
│    • Rol: Asistente Clínico de Urgencias (Graph RAG)                │
│    • Caso activo (edad, sexo, vitales, síntomas, severidad)         │
│    • Ground Truth del grafo (precauciones, fármacos, dietas)        │
│    • Reglas bioéticas estrictas:                                    │
│      - Respetar severidad del clasificador determinista             │
│      - Priorizar atención presencial si es Severo                   │
│      - Prohibido inventar dosis o tratamientos                      │
│      - Tono médico sobrio, empático, en español                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 4: INFERENCIA CONVERSACIONAL Y STREAMING                      │
│                                                                     │
│  INPUT:  Pregunta del usuario (lenguaje natural o sugerencia)       │
│                          ↓                                          │
│  CONEXIÓN HÍBRIDA A OLLAMA:                                         │
│    • Túnel remoto: https://[túnel].loca.lt                          │
│    • Fallback local: http://localhost:11434                         │
│                          ↓                                          │
│  GENERACIÓN: temperature = 0.3 (alta fidelidad)                     │
│                          ↓                                          │
│  OUTPUT: Respuesta token a token vía st.write_stream()              │
│                                                                     │
│  MODO RESPALDO (Zero-Failure):                                      │
│    Si Ollama está desconectado → generar_respuesta_fallback_grafo() │
│    entrega directamente los hechos de la ontología                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 Esquema del Grafo de Conocimiento

**Nodos:**

- 🔵 **Azul (`#3b82f6`):** `Disease` (41 condiciones clínicas)
- 🔴 **Rojo (`#ef4444`):** `Severity` (Severo, Moderado, Leve)
- 🟣 **Púrpura (`#8b5cf6`):** `Precaution` (Medidas preventivas)
- 🌸 **Rosa (`#ec4899`):** `Medication` (Familias farmacológicas de soporte)
- 🟢 **Verde (`#10b981`):** `Diet` (Pautas nutricionales e hidroelectrolíticas)

**Relaciones:**

- `(Symptom)-[:INDICATES]->(Disease)`
- `(Disease)-[:HAS_SEVERITY]->(Severity)`
- `(Disease)-[:REQUIRES_PRECAUTION]->(Precaution)`
- `(Disease)-[:RECOMMENDS_MEDICATION]->(Medication)`
- `(Disease)-[:SUGGESTS_DIET]->(Diet)`

**Estructura de exportación:**

```
data/knowledge_graph/
├── medical_knowledge_graph.graphml                 # XML estándar (Gephi, Cytoscape)
├── medical_knowledge_graph.json                    # Node-Link (D3.js)
├── medical_knowledge_graph.pkl                     # Pickle (carga < 2 ms)
├── medical_knowledge_graph_fruchterman_reingold.png # Layout estático 300 DPI
└── medical_knowledge_graph_interactive.html        # Vis.js interactivo
```

---

### 4.4 Componentes de la Interfaz Streamlit

**Pestaña 1: "Triaje en Vivo"**

- `st.number_input("Edad", min_value=0, max_value=120, value=30)`
- `st.selectbox("Género", ["Masculino", "Femenino", "Otro"])`
- `st.multiselect("Síntomas", opciones_sintomas)` (hasta 3)
- `st.number_input("Frecuencia Cardíaca (bpm)", 30-220, 80)`
- `st.number_input("Temperatura (°C)", 34.0-42.0, 37.0, step=0.1)`
- `st.number_input("Presión Sistólica (mmHg)", 70-250, 120)`
- `st.number_input("Presión Diastólica (mmHg)", 40-150, 80)`
- `st.number_input("Saturación O₂ (%)", 70-100, 98)`
- `st.button("Evaluar Prioridad")`

**Resultado visual:**

- 🔴 **SEVERO** → "Atención inmediata requerida"
- 🟡 **MODERADO** → "Atención en menos de 60 minutos"
- 🟢 **LEVE** → "Atención no urgente"
- `st.progress()` con probabilidad por clase

**Pestaña 2: "Chatbot Clínico"**

- `st.chat_input()` para consulta libre
- Sugerencias rápidas: *"¿Cuáles son los signos de alarma?"*
- `st.write_stream()` para respuesta token a token
- Tarjeta de contexto del paciente (`.patient-context-card`)
- Diseño adaptable modo claro/oscuro

**Pestaña 3: "Visualización del Grafo"**

- Visor interactivo del grafo de conocimiento
- Exploración de relaciones Síntoma → Enfermedad → Recomendación

**Pestaña 4: "Disclaimer"**

> "Esta herramienta es solo para fines educativos y de investigación. No constituye diagnóstico médico. Las recomendaciones provienen de una base de conocimiento predefinida, no de generación libre del modelo."

---

## 5. Resultados Obtenidos (Demo)

### 5.1 Métricas del Modelo

| Métrica                       | Valor                    |
| ------------------------------ | ------------------------ |
| **Accuracy (Test)**      | **100.0%**         |
| **Macro F1-Score**       | **100.0%**         |
| **Sensibilidad Severe**  | **100.0%** (76/76) |
| **Especificidad**        | **100.0%**         |
| **Tiempo de inferencia** | < 1 segundo              |

### 5.2 Funcionalidades Demostrables

1. **Formulario de triaje:** El usuario ingresa datos del paciente y obtiene la clasificación de severidad con código de color semáforo.
2. **Probabilidades calibradas:** Se muestra la certeza del modelo para cada clase (Mild, Moderate, Severe).
3. **Chatbot clínico anclado:** El usuario puede preguntar en lenguaje natural y recibe respuestas basadas en el grafo de conocimiento, sin alucinaciones.
4. **Visualización del grafo:** Exploración interactiva de las relaciones clínicas.
5. **Modo respaldo:** Si el LLM no está disponible, el sistema entrega directamente los hechos de la ontología.

### 5.3 Casos de Prueba

**Caso 1 — Paciente leve:**

- Entrada: Edad 30, FC 75, Temp 37.0°C, SatO₂ 98%, síntomas: runny nose, sore throat
- Predicción: **Mild (Leve)** → "Atención no urgente"
- Recomendación: Reposo e hidratación

**Caso 2 — Paciente severo:**

- Entrada: Edad 65, FC 115, Temp 38.5°C, SatO₂ 91%, síntomas: cough, shortness of breath, fever
- Predicción: **Severe (Severo)** → "Atención inmediata requerida"
- Recomendación: Hospitalización y medicación

---

## 6. Conclusiones y Reflexiones

### 6.1 Conclusiones

1. **Sobre los outliers:** Univariadamente el dataset no contiene valores atípicos convencionales porque las variables fueron muestreadas de distribuciones uniformes con límites prefijados (curtosis ≈ -1.2). Sin embargo, el análisis multivariable identificó **156 inconsistencias biológicas** (PAD ≥ PAS) que deben considerarse al interpretar la reproducibilidad en escenarios clínicos reales.
2. **Rendimiento y seguridad en triaje:** El modelo Random Forest demostró un **100% de sensibilidad en casos severos**, mitigando el riesgo primario de un servicio de urgencias: la infraclasificación de pacientes críticos.
3. **Valor de la ingeniería de características:** Las variables derivadas (Shock Index, PAM, banderas clínicas) explican más del **27% de la capacidad de decisión del modelo**, demostrando que transformar variables crudas en indicadores biomédicos mejora sustancialmente el rendimiento.
4. **Graph RAG como anclaje de seguridad:** La arquitectura híbrida (ML + Grafo + LLM) garantiza que las recomendaciones **no sean inventadas por el LLM**, sino recuperadas de una fuente de verdad estructurada y verificable. Esto resuelve el problema crítico de la alucinación en aplicaciones clínicas.
5. **Despliegue exitoso:** La aplicación Streamlit desplegada en Streamlit Community Cloud es **accesible públicamente**, interactiva y demuestra el flujo completo de triaje con chatbot clínico anclado.

### 6.2 Reflexiones sobre el Proceso

- **Sobre la naturaleza sintética del dataset:** La ausencia de outliers univariados y la presencia de inconsistencias fisiológicas multivariadas revelan que el dataset fue generado sintéticamente. Esto es una **limitación importante** que debe considerarse al interpretar los resultados: un modelo entrenado con datos sintéticos puede no generalizar a datos clínicos reales sin validación adicional.
- **Sobre la ética en IA médica:** El proyecto refuerza un principio fundamental: **el LLM nunca diagnostica ni decide la urgencia por sí solo**. El modelo de ML (entrenado con datos) y el grafo (conocimiento curado) toman las decisiones, y el LLM solo comunica. Esta separación de responsabilidades es esencial para cualquier aplicación clínica responsable.
- **Sobre la reproducibilidad:** El uso de un dataset de código abierto y la publicación del código en GitHub garantizan que el proyecto sea **reproducible y auditable**, cumpliendo con los principios de ciencia abierta.

### 6.3 Limitaciones

1. **Dataset sintético/simplificado:** Los datos no provienen de pacientes reales.
2. **No reemplaza el juicio clínico profesional:** Es una herramienta de apoyo, no un sistema de diagnóstico autónomo.
3. **Requiere validación con datos reales** antes de cualquier uso clínico.
4. **Datos no ecuatorianos:** El dataset es de código abierto, no de Ecuador.
5. **Riesgo de sesgo:** El dataset puede no representar la diversidad demográfica de la población ecuatoriana.

### 6.4 Futuras Mejoras

1. **Integración con chatbot conversacional completo:** Implementar diálogo multi-turno con memoria de contexto (previsto para el examen final).
2. **Validación con datos reales:** Colaborar con hospitales ecuatorianos para validar el modelo con datos clínicos anonimizados.
3. **Expansión del grafo de conocimiento:** Incorporar más enfermedades, síntomas y tratamientos validados por literatura médica.
4. **Integración con LLM locales:** Usar Ollama con modelos como Llama 3 o Mistral para inferencia completamente local y privada.
5. **Implementación de Graph RAG con Neo4j:** Migrar el grafo de NetworkX a Neo4j para consultas Cypher más eficientes y escalables.
6. **Predicción de tratamiento:** Agregar un segundo modelo que prediga `Treatment_Plan` basado en el diagnóstico y severidad.
7. **Sistema de alertas en tiempo real:** Integrar con sistemas de historia clínica electrónica para triaje automatizado en admisión.
8. **Auditoría de sesgo:** Evaluar el modelo con datos demográficamente diversos para identificar y mitigar sesgos.

---

## 7. Entregables

| Entregable                          | Descripción                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| **Notebook de entrenamiento** | Script`.ipynb` con preprocesamiento, EDA, benchmark y exportación del modelo |
| **Archivo del modelo**        | `triaje_model.joblib` (Random Forest entrenado)                               |
| **Grafo de conocimiento**     | `medical_knowledge_graph.pkl` (347 nodos, 575 aristas)                        |
| **App Streamlit**             | `app.py` con formulario de triaje + chatbot clínico                          |
| **requirements.txt**          | Dependencias: streamlit, pandas, scikit-learn, joblib, numpy, networkx, groq    |
| **Enlace público**           | URL de Streamlit Community Cloud                                                |
| **Presentación**             | 6 diapositivas (máximo)                                                        |
| **Informe técnico**          | Documento de análisis bioestadístico y benchmark de modelos                   |

---

## 8. Criterios de Éxito (Alineados con la Rúbrica)

| Criterio                                            | Peso | Cómo se cumple                                                                    |
| --------------------------------------------------- | ---- | ---------------------------------------------------------------------------------- |
| **Definición y Justificación del Problema** | 15%  | Secciones 1 y 2; triaje es problema real con evidencia clínica                    |
| **Uso de Herramientas del Módulo**           | 30%  | Random Forest, preprocesamiento, clasificación multiclase, Graph RAG, LLM         |
| **Implementación y Resultados**              | 35%  | App desplegada en Streamlit Cloud con enlace público, 100% sensibilidad en Severe |
| **Presentación (6 diapositivas)**            | 10%  | Estructura: Problema → Justificación → Solución → Demo → Conclusiones        |
| **Conclusiones y Futuras Mejoras**            | 10%  | Sección 6: limitaciones, ética, trabajo futuro                                   |

---

## Resumen Ejecutivo

**Proyecto:** Sistema de triaje inteligente que predice la gravedad de un paciente (Mild/Moderate/Severe) a partir de síntomas y signos vitales, con chatbot clínico anclado a un grafo de conocimiento (Graph RAG).

**Dataset:** `bouramah/projet_data_mining_sante` (GitHub) — 2,000 admisiones hospitalarias con 22 variables enriquecidas.

**Modelo:** Random Forest (200 estimadores) — 100% sensibilidad en casos severos, 100% accuracy en test.

**Arquitectura:** Híbrida ML + Grafo de conocimiento (NetworkX, 347 nodos, 575 aristas) + LLM (Ollama/Groq) con anclaje bioético.

**App:** Streamlit con formulario de triaje, chatbot clínico y visualización del grafo.

**Despliegue:** Streamlit Community Cloud vía GitHub.

**Valor:** Acelera la clasificación inicial de pacientes, reduce carga cognitiva del personal de triaje, y puede implementarse como sistema de pre-triaje digital en contextos con recursos limitados.

---

*Documento final del proyecto — Septiembre 2026*
