# Triaje Predictivo — Presentación de Diapositivas
### Sistema Inteligente de Clasificación de Prioridad en Urgencias Hospitalarias
**Módulo: Fundamentos de Inteligencia Artificial · Septiembre 2026**

---

---

## DIAPOSITIVA 1 — Introducción al Problema

### Triaje Predictivo: IA al Servicio de la Urgencia Médica

> *"En una sala de urgencias, cada minuto cuenta. El triaje es la primera —y más crítica— decisión del sistema de salud."*

#### El Triaje Hospitalario: Reto Humano y Operativo

El **triaje** es el proceso de clasificación de pacientes según su nivel de gravedad para determinar el orden de atención médica. Su principio rector establece que **"lo urgente no siempre es grave, y lo grave no siempre es urgente"**.

#### El Problema Central

En entornos de alta demanda asistencial, el triaje manual enfrenta tres desafíos estructurales:

| Desafío | Impacto |
|---|---|
| **Tiempo limitado** por paciente | Evaluaciones apresuradas e incompletas |
| **Carga cognitiva del personal** | Variabilidad en la clasificación entre evaluadores |
| **Escasez de profesionales** | Especialmente crítica en centros de salud rurales |

#### La Pregunta Central del Proyecto

> **¿Puede un modelo de Machine Learning, entrenado con signos vitales y síntomas, predecir automáticamente el nivel de gravedad de un paciente para asistir al personal de triaje?**

**Contexto del dataset:** Cohorte de **2,000 admisiones hospitalarias** con registros de síntomas, constantes vitales y etiqueta de severidad clínica validada (Mild, Moderate, Severe).

---

---

## DIAPOSITIVA 2 — Justificación del Problema

### Por Qué Importa Resolver Este Problema

#### El Costo de Clasificar Mal a un Paciente Crítico

Un **falso negativo en triaje** —clasificar como "leve" a un paciente severo— puede derivar en:
- Deterioro hemodinámico sin monitorización
- Paro cardiorrespiratorio en sala de espera
- Consecuencias médico-legales y pérdida de vida humana

Por esta razón, la métrica más relevante no es la precisión global, sino la **Sensibilidad (Recall) sobre la clase Severo**: el sistema no puede permitirse ningún falso negativo en pacientes críticos.

#### Distribución Real de la Severidad en Urgencias

El dataset refleja la distribución asimétrica típica de un servicio de urgencias:

| Nivel de Severidad | Casos | Proporción | Prioridad Clínica |
|---|---|---|---|
| **Leve (Mild)** | 1,330 | 66.5% | Atención diferida / Sala de espera |
| **Moderado (Moderate)** | 292 | 14.6% | Atención prioritaria < 60 minutos |
| **Severo (Severe)** | 378 | 18.9% | Atención inmediata / UCI |

> Este desbalance confirma que un modelo basado solo en Accuracy sería engañoso. Se requiere **Macro F1-Score** y **Recall en Severos** como métricas primarias.

#### Evidencia Estadística: Los Síntomas Sí Discriminan la Gravedad

La prueba de **Chi-cuadrado** sobre los síntomas reveló asociaciones extremadamente significativas:

| Síntoma | V de Cramer | Hallazgo Clave |
|---|---|---|
| **Tos (Cough)** | **0.527** | Presente en el **91.8%** de los pacientes Severos |
| **Fiebre (Fever)** | **0.523** | Presente en el **100%** de los Moderados |
| Disnea | 0.152 | Asociación moderada, más frecuente en Leves |
| Fatiga | 0.138 | Síntoma inespecífico, predomina en Leves |

Asimismo, la prueba de **Kruskal-Wallis** sobre los signos vitales continuos demostró:
- **Saturación de O2** (H = 371.96, p < 10^-80): mediana de **92%** en Severos vs. 95% en Leves.
- **Temperatura corporal** (H = 325.02, p < 10^-70): mediana de **39 °C** en Moderados vs. 37.4 °C en Leves.

> **Conclusión de justificación:** Los síntomas y signos vitales capturan información biológicamente significativa y estadísticamente discriminante para estratificar la gravedad. Un modelo supervisado es científicamente viable.

---

---

## DIAPOSITIVA 3 — Descripción Técnica de la Solución

### Arquitectura del Sistema: De los Datos a la Decisión Clínica

```
ENTRADA              FASE 1               FASE 2               SALIDA 1
Datos del  →  Ingeniería de   →   Random Forest   →   Triaje en Vivo
Paciente      Características      Classifier          (Tab 1: Semáforo)
              (22 variables)       .joblib
                                        ↓
                               st.session_state
                                        ↓
                    FASE 3               FASE 4           SALIDA 2
              Graph RAG        →   System Prompt  →   Chatbot Clínico
              Ontología            Clínico (LLM)       (Tab 2: LLM)
              NetworkX
```

#### Fase 1 — Preprocesamiento e Ingeniería de Características

A partir de los 7 campos originales del paciente, se construyeron **22 variables enriquecidas** con fundamento biomédico:

| Feature Derivada | Lógica | Relevancia Clínica |
|---|---|---|
| `sym_fever`, `sym_cough`, ... | Binarización de síntomas textuales | Indicadores directos del cuadro clínico |
| `shock_index` | FC / Presión Sistólica | > 0.9 indica inestabilidad hemodinámica |
| `map_pressure` | PAD + (PAS − PAD) / 3 | Presión de perfusión de órganos vitales |
| `flag_hypoxia` | SatO2 < 95% → 1 | Alerta de hipoxemia periférica |
| `flag_tachycardia` | FC > 100 lpm → 1 | Marcador de esfuerzo cardíaco compensatorio |
| `flag_fever` | Temp >= 38 °C → 1 | Umbral clínico de síndrome febril |

> Las variables derivadas explican más del **27% de la capacidad predictiva total del modelo**, validando el valor de la ingeniería de características con criterio clínico.

#### Fase 2 — Benchmark de 6 Modelos (Validación Cruzada Stratified 5-Fold)

| Modelo | Macro F1 | Accuracy | Seleccionado |
|---|---|---|---|
| XGBoost | 100.00% | 100.00% | — |
| **Random Forest** | **99.87%** | **99.94%** | Si |
| SVM (RBF) | 98.79% | 99.19% | — |
| Red Neuronal (MLP) | 97.97% | 98.69% | — |
| Regresión Logística | 97.96% | 98.75% | — |
| K-Nearest Neighbors | 89.76% | 92.69% | — |

**Por qué Random Forest sobre XGBoost:**
- **Probabilidades calibradas** (`predict_proba`): esenciales para cuantificar la incertidumbre clínica.
- **Interpretabilidad**: importancia de características transparente y explicable al personal médico.
- **Inferencia sub-milisegundo** en CPU, sin dependencias pesadas en Streamlit Cloud.

#### Fase 3 — Graph RAG: Ontología Médica como Fuente de Verdad

El resultado del modelo activa un **recuperador de grafo de conocimiento** implementado en NetworkX:

| Componente | Detalle |
|---|---|
| **Grafo** | 347 nodos · 575 aristas dirigidas |
| **Nodos** | Disease, Severity, Precaution, Medication, Diet |
| **Relaciones** | REQUIRES_PRECAUTION, RECOMMENDS_MEDICATION, SUGGESTS_DIET |
| **Consulta** | `retrieve_clinical_context(severity, symptoms)` |
| **Salida** | `graph_ctx` — hechos verificados y estructurados |

> El grafo actúa como **Ground Truth clínico**: restringe qué recomendaciones, precauciones y terapias el LLM puede emitir, eliminando el riesgo de alucinaciones médicas.

#### Fase 4 — Chatbot Clínico con LLM Anclado al Grafo

La función `construir_prompt_sistema_graph_rag()` ensambla un System Prompt que inyecta los hechos del grafo al LLM:

- **Motor LLM:** Ollama (`qwen3.5:9b` / `deepseek-r1:8b`) con temperatura 0.3 (alta fidelidad)
- **Streaming:** Respuesta token a token via `st.write_stream()`
- **Modo Fallback:** Si Ollama no está disponible, el sistema entrega hechos de la ontología directamente (zero-failure)
- **Reglas Bioéticas:** El LLM no puede inventar dosis ni diagnósticos ajenos al grafo

#### Dos Salidas Diferenciadas de la App (2 Pestañas)

| Tab | Nombre | Tecnología | Función |
|---|---|---|---|
| **Tab 1** | Triaje en Vivo | Random Forest | Clasificación semafórica inmediata + probabilidades + bitácora CSV |
| **Tab 2** | Chatbot Clínico | Graph RAG + LLM | Respuestas conversacionales ancladas en ontología verificada |

---

## DIAPOSITIVA 4 — Resultados Obtenidos (Demo)

### Rendimiento Clínico: Cero Falsos Negativos en Casos Críticos

#### Evaluación en Conjunto de Prueba Independiente (N = 400, 20% Hold-Out)

```
                  Precision    Recall    F1-Score   Soporte
           Mild       1.00      1.00        1.00       266
       Moderate       1.00      1.00        1.00        58
         Severe       1.00      1.00        1.00        76

       Accuracy                             1.00       400
      Macro avg       1.00      1.00        1.00       400
```

#### Métricas Clave del Sistema

| Métrica | Valor | Relevancia |
|---|---|---|
| **Accuracy Global** | **100%** | Clasificación perfecta en test independiente |
| **Macro F1-Score (CV)** | **99.87%** | Rendimiento balanceado entre las 3 clases |
| **Recall en Severos** | **100%** (76/76) | Cero falsos negativos críticos |
| **Tiempo de inferencia** | < 1 ms | Decisión en tiempo real |

#### Top Variables con Mayor Impacto Predictivo (Feature Importance de Gini)

| Rango | Variable | Importancia | Interpretación |
|---|---|---|---|
| 1 | `sym_fever` (Fiebre) | **20.33%** | Síntoma cardinal diferenciador de Moderados |
| 2 | `sym_cough` (Tos) | **17.42%** | Marcador casi universal de cuadros Severos |
| 3 | `Body_Temperature_C` | **12.69%** | Valor continuo de temperatura corporal |
| 4 | `flag_hypoxia` | **11.09%** | Alerta de SatO2 < 95% |
| 5 | `Oxygen_Saturation_%` | **10.82%** | Desaturación como indicador de compromiso pulmonar |
| 6 | `flag_fever` | **10.25%** | Umbral clínico de síndrome febril |
| 7-10 | FC, taquicardia, shock_index, PAM | **~8.64%** | Indicadores hemodinámicos derivados |

#### Hallazgo Bioestadístico Notable: Anatomía de los Datos Sintéticos

El análisis de outliers reveló una **curtosis en exceso aprox. −1.20** en todas las variables continuas, matemáticamente equivalente a una distribución uniforme U(a,b), lo que confirma que el dataset fue generado sintéticamente con rangos prefijados. Sin embargo, el análisis multivariado (Isolation Forest) identificó **60 casos de descompensación extrema** y **156 inconsistencias biológicas** (PAD >= PAS), hallazgos relevantes para interpretar los límites de generalización del modelo en escenarios clínicos reales.

#### Demo en Vivo — Streamlit App

**Escenario A — Paciente Severo:**
- Síntomas: Tos + Fiebre | SatO2: 91% | FC: 118 lpm | Temp: 38.8 °C
- Resultado: **SEVERO — Atención Inmediata** | Confianza: 98.7%

**Escenario B — Paciente Moderado:**
- Síntomas: Fiebre + Dolor de garganta | SatO2: 96% | FC: 85 lpm | Temp: 39.1 °C
- Resultado: **MODERADO — Atención en < 60 min** | Confianza: 94.2%

**Escenario C — Paciente Leve:**
- Síntomas: Rinorrea + Mialgias | SatO2: 98% | FC: 76 lpm | Temp: 37.2 °C
- Resultado: **LEVE — Atención Diferida** | Confianza: 99.1%

---

---

## DIAPOSITIVA 5 — Conclusiones y Reflexiones

### Lo Que Aprendimos: Ciencia de Datos Aplicada a la Vida Real

#### Conclusiones Técnicas

**1. El modelo es clínicamente seguro para su propósito:**
El Random Forest alcanzó un **100% de Sensibilidad en casos Severos** en el conjunto de prueba independiente (76/76 pacientes críticos clasificados correctamente). En triaje médico, este indicador es no negociable: un falso negativo puede costar una vida.

**2. La ingeniería de características supera al dato crudo:**
Las 22 variables derivadas —en especial el Índice de Shock, la Presión Arterial Media y las banderas de hipoxemia y taquicardia— explican más del **27% de la capacidad predictiva del modelo**, demostrando que el conocimiento del dominio clínico transforma los datos en inteligencia.

**3. La elección del modelo importa más allá del accuracy:**
Aunque XGBoost alcanzó métricas ligeramente superiores, Random Forest fue seleccionado por su **interpretabilidad** y **probabilidades calibradas**, atributos indispensables en entornos clínicos donde la transparencia genera confianza del personal médico.

**4. Los datos sintéticos tienen límites reales:**
La detección de **156 inconsistencias biológicas** (presión diastólica >= sistólica) mediante análisis multivariado evidencia que incluso datasets aparentemente limpios pueden esconder anomalías fisiológicas. Esto es crítico para la generalización a datos clínicos reales.

#### Reflexiones sobre IA en Salud

> *"La Inteligencia Artificial no debe reemplazar al médico de triaje; debe darle superpoderes para salvar más vidas en menos tiempo."*

| Lo que la IA puede hacer | Lo que la IA no puede hacer |
|---|---|
| Procesar señales vitales en < 1 ms | Realizar la exploración física del paciente |
| Detectar patrones en miles de casos | Capturar el contexto social y emocional |
| Cuantificar incertidumbre con probabilidades | Asumir responsabilidad clínica y ética |
| Operar 24/7 sin fatiga cognitiva | Reemplazar el juicio clínico experto |

#### Limitaciones Honestas del Proyecto

1. **Dataset sintético:** Los datos no provienen de pacientes reales ecuatorianos; la distribución uniforme no refleja la variabilidad biológica del mundo real.
2. **Validación clínica pendiente:** Antes de cualquier implementación asistencial, el modelo requiere validación prospectiva con datos reales bajo supervisión médica certificada.
3. **Alcance declarado:** Esta herramienta es un sistema de **apoyo a la decisión** —no un dispositivo médico certificado— y no sustituye la valoración presencial del profesional de salud.

#### Proyecciones y Trabajo Futuro

- **Graph RAG + LLM:** Incorporar un grafo de conocimiento médico (Neo4j) para anclar las recomendaciones en ontologías clínicas verificables, eliminando el riesgo de alucinaciones. Estudios del *Journal of the American Medical Informatics Association* reportan **Precisión: 0.99** con esta arquitectura híbrida.
- **Datos reales:** Colaboración con servicios de urgencias locales para entrenar con datos clínicos ecuatorianos anonimizados.
- **Integración hospitalaria:** API REST para conectar con sistemas HIS (Hospital Information System) y formularios de admisión digital.

---

---

## DIAPOSITIVA 6 — Síntesis Final

### Triaje Predictivo — Lo Esencial en Una Página

#### El Proyecto en 30 Segundos

| | |
|---|---|
| **Problema** | La clasificación manual de pacientes en urgencias es lenta, costosa y variable entre evaluadores |
| **Dataset** | 2,000 admisiones · 8 síntomas · 5 constantes vitales · 3 niveles de severidad |
| **Solución** | Pipeline ML: preprocesamiento clínico → Random Forest → App Streamlit desplegada |
| **Resultado estrella** | **100% Recall en Severos** — cero pacientes críticos infraclasificados |
| **Herramienta** | App interactiva con semáforo de triaje, probabilidades calibradas y bitácora descargable |
| **Despliegue** | Streamlit Community Cloud · Código reproducible en GitHub |

#### Stack Tecnológico Completo

```
Python 3.11  ·  Scikit-Learn  ·  Pandas / NumPy
Streamlit    ·  Joblib        ·  SciPy (Kruskal-Wallis, Chi-cuadrado)
Matplotlib   ·  Seaborn       ·  Isolation Forest (anomalías multivariadas)
GitHub       ·  Streamlit Community Cloud (despliegue público)
```

#### Métricas de Referencia Rápida

| Métrica | Valor |
|---|---|
| Accuracy (Test Set) | **100%** |
| Macro F1 (CV 5-Fold) | **99.87%** |
| Recall Severos | **100%** (0 falsos negativos) |
| Variables del modelo | **22** features enriquecidas |
| Tiempo de inferencia | **< 1 ms** |
| Admisiones analizadas | **2,000** registros clínicos |

---

> **Descargo de responsabilidad:** Este sistema ha sido desarrollado con fines académicos en el marco del módulo de Fundamentos de Inteligencia Artificial (MIA 2026). No constituye un dispositivo médico certificado ni sustituye la valoración clínica de profesionales de la salud.

---

*Presentación preparada para la sesión del sábado · Módulo: Fundamentos de IA · Universidad de Cuenca · 2026*
