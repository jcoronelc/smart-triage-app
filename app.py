"""
Aplicación Web de Triaje Predictivo en Urgencias
Desarrollada con Streamlit y Scikit-Learn
Proyecto: Sistema de Apoyo a la Clasificación de Prioridad Clínica
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Asegurar visibilidad del módulo src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preprocessing import (
    ClinicalFeatureExtractor,
    LISTA_SINTOMAS_CANONICOS,
    TRADUCCION_SINTOMAS,
    TRADUCCION_INVERSA_SINTOMAS
)

# Configuración de página
st.set_page_config(
    page_title="Triaje Predictivo | Urgencias",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS profesionales para estética clínica sobria
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a2e40;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .triage-card {
        padding: 1.4rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .triage-severe {
        background-color: #fef2f2;
        border-left: 6px solid #b91c1c;
        border: 1px solid #fecaca;
    }
    .triage-moderate {
        background-color: #fffbeb;
        border-left: 6px solid #d97706;
        border: 1px solid #fde68a;
    }
    .triage-mild {
        background-color: #f0fdf4;
        border-left: 6px solid #15803d;
        border: 1px solid #bbf7d0;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-danger { background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-warning { background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .badge-success { background-color: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .badge-info { background-color: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
</style>
""", unsafe_allow_html=True)

# 1. Carga del Modelo en Caché
@st.cache_resource(show_spinner="Cargando pipeline del modelo...")
def cargar_pipeline_triaje():
    model_path = os.path.join(BASE_DIR, 'models', 'triaje_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Archivo de modelo no encontrado en: {model_path}. Ejecute src/train_and_evaluate.py primero.")
        st.stop()
    return joblib.load(model_path)

pipeline = cargar_pipeline_triaje()

# Inicialización de historial en sesión
if 'historial_triaje' not in st.session_state:
    st.session_state.historial_triaje = []

# Inicialización reactiva de variables de entrada para sincronización con widgets
if 'age_key' not in st.session_state: st.session_state['age_key'] = 45
if 'gender_key' not in st.session_state: st.session_state['gender_key'] = "Femenino"
if 'hr_key' not in st.session_state: st.session_state['hr_key'] = 80
if 'temp_key' not in st.session_state: st.session_state['temp_key'] = 36.8
if 'sys_key' not in st.session_state: st.session_state['sys_key'] = 120
if 'dia_key' not in st.session_state: st.session_state['dia_key'] = 80
if 'sat_key' not in st.session_state: st.session_state['sat_key'] = 98
if 'symptoms_key' not in st.session_state: st.session_state['symptoms_key'] = ["Dolor de garganta", "Congestión / Rinorrea"]

# SIDEBAR: Información del Modelo y Disclaimer
with st.sidebar:
    st.title("Triaje Predictivo")
    st.caption("Sistema de Apoyo a la Decisión Clínica (CDSS)")
    
    st.divider()
    st.markdown("### Especificaciones del Modelo")
    st.markdown("""
    - **Algoritmo:** Random Forest Classifier (200 estimadores)
    - **Clases del Dataset:** `Mild`, `Moderate`, `Severe`
    - **Validación:** 5-Fold Stratified CV + Hold-out Test ($N=400$)
    - **Macro F1-Score:** 100.0%
    - **Sensibilidad (Severe):** 100.0%
    - **Variables analizadas:** 22 (signos vitales, síntomas, PAM, Shock Index)
    """)
    
    st.divider()
    st.markdown("### Clasificación de Severidad")
    st.markdown("""
    - **Severe (Severo):** Atención médica inmediata. Riesgo de descompensación hemodinámica o respiratoria.
    - **Moderate (Moderado):** Atención prioritaria sugerida en menos de 60 min. Cuadro agudo febril/sistémico.
    - **Mild (Leve):** Atención ambulatoria estándar / sala de espera general.
    """)
    
    st.divider()
    st.info("""
    **Descargo de Responsabilidad:**  
    Herramienta prototipo desarrollada con fines estrictamente académicos e investigativos. No reemplaza el juicio clínico médico profesional ni la evaluación presencial de enfermería.
    """)

# HEADER PRINCIPAL
st.markdown('<div class="main-header">Sistema de Triaje Predictivo en Urgencias</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clasificación asistida de gravedad clínica basada en anamnesis inmediata, constantes vitales y perfiles hemodinámicos.</div>', unsafe_allow_html=True)

# Pestañas Principales
tab_triaje, tab_analitica = st.tabs(["Triaje en Vivo", "Análisis Estadístico y Modelos"])

with tab_triaje:
    # Función callback para actualizar los widgets inmediatamente
    def cargar_caso(caso):
        if caso == "severe":
            st.session_state['age_key'] = 68
            st.session_state['gender_key'] = "Masculino"
            st.session_state['hr_key'] = 115
            st.session_state['temp_key'] = 37.8
            st.session_state['sys_key'] = 145
            st.session_state['dia_key'] = 85
            st.session_state['sat_key'] = 91
            st.session_state['symptoms_key'] = ["Tos persistente", "Dificultad respiratoria (Disnea)", "Fatiga / Astenia"]
        elif caso == "moderate":
            st.session_state['age_key'] = 35
            st.session_state['gender_key'] = "Femenino"
            st.session_state['hr_key'] = 92
            st.session_state['temp_key'] = 39.2
            st.session_state['sys_key'] = 120
            st.session_state['dia_key'] = 80
            st.session_state['sat_key'] = 97
            st.session_state['symptoms_key'] = ["Fiebre", "Cefalea (dolor de cabeza)", "Dolor corporal"]
        elif caso == "mild":
            st.session_state['age_key'] = 25
            st.session_state['gender_key'] = "Femenino"
            st.session_state['hr_key'] = 72
            st.session_state['temp_key'] = 36.6
            st.session_state['sys_key'] = 115
            st.session_state['dia_key'] = 75
            st.session_state['sat_key'] = 99
            st.session_state['symptoms_key'] = ["Congestión / Rinorrea", "Dolor de garganta"]

    # BOTONES DE PRESETS
    st.markdown("##### Cargar Casos Clínicos Tipo:")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.button("Caso Severo (Severe)", on_click=cargar_caso, args=("severe",), use_container_width=True)
    with col_p2:
        st.button("Caso Moderado (Moderate)", on_click=cargar_caso, args=("moderate",), use_container_width=True)
    with col_p3:
        st.button("Caso Leve (Mild)", on_click=cargar_caso, args=("mild",), use_container_width=True)

    st.divider()

    # FORMULARIO DE ADMISIÓN
    col_left, col_right = st.columns([1.1, 1.2], gap="large")

    with col_left:
        st.subheader("Datos Clínicos del Paciente")
        
        with st.expander("1. Datos Demográficos", expanded=True):
            c_dem1, c_dem2 = st.columns(2)
            with c_dem1:
                age = st.number_input("Edad (años)", min_value=1, max_value=110, step=1, key="age_key")
            with c_dem2:
                gender = st.selectbox("Sexo biológico", ["Femenino", "Masculino", "Otro"], key="gender_key")

        with st.expander("2. Síntomas Referidos en Admisión", expanded=True):
            opciones_sintomas_es = list(TRADUCCION_SINTOMAS.values())
            selected_symptoms_es = st.multiselect(
                "Seleccione los síntomas manifestados (1 a 3 principales):",
                options=opciones_sintomas_es,
                key="symptoms_key",
                help="Seleccione hasta 3 síntomas cardinales descritos por el paciente."
            )
            if len(selected_symptoms_es) == 0:
                st.caption("Sin síntomas seleccionados se asume paciente asintomático.")

        with st.expander("3. Constantes y Signos Vitales", expanded=True):
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                hr = st.number_input("Frecuencia Cardíaca (lpm)", min_value=30, max_value=220, step=1, key="hr_key")
                temp = st.number_input("Temperatura Corporal (°C)", min_value=34.0, max_value=43.0, step=0.1, format="%.1f", key="temp_key")
                sat = st.slider("Saturación O₂ (%)", min_value=70, max_value=100, step=1, key="sat_key")
            with c_v2:
                sys_bp = st.number_input("Presión Sistólica (mmHg)", min_value=60, max_value=240, step=1, key="sys_key")
                dia_bp = st.number_input("Presión Diastólica (mmHg)", min_value=35, max_value=140, step=1, key="dia_key")
                
                # Cálculo de índices hemodinámicos en tiempo real
                shock_idx = round(hr / (sys_bp if sys_bp > 0 else 120), 2)
                pam = round(dia_bp + (sys_bp - dia_bp) / 3.0, 1)
                st.metric("Índice de Shock (FC / PAS)", f"{shock_idx}", delta="Inestabilidad > 0.9" if shock_idx > 0.9 else "Normal (0.5 - 0.7)", delta_color="inverse" if shock_idx > 0.9 else "normal")
                st.metric("Presión Arterial Media (PAM)", f"{pam} mmHg", help="Meta clínica en urgencias: >= 65 mmHg")

        evaluar_btn = st.button("Registrar Paciente en Bitácora de Triaje", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Resultado de la Clasificación")
        
        # Preparación de datos para el pipeline
        sintomas_en = [TRADUCCION_INVERSA_SINTOMAS[s] for s in selected_symptoms_es if s in TRADUCCION_INVERSA_SINTOMAS]
        s1 = sintomas_en[0] if len(sintomas_en) > 0 else "None"
        s2 = sintomas_en[1] if len(sintomas_en) > 1 else "None"
        s3 = sintomas_en[2] if len(sintomas_en) > 2 else "None"
        
        gender_en = "Male" if gender == "Masculino" else "Female"
        
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender_en,
            'Heart_Rate_bpm': hr,
            'Body_Temperature_C': temp,
            'Blood_Pressure_mmHg': f"{sys_bp}/{dia_bp}",
            'Oxygen_Saturation_%': sat,
            'Symptom_1': s1,
            'Symptom_2': s2,
            'Symptom_3': s3
        }])
        
        # Inferencia
        pred_class = pipeline.predict(input_df)[0]
        pred_probas = pipeline.predict_proba(input_df)[0]
        classes = list(pipeline.classes_)
        prob_dict = {c: float(p) for c, p in zip(classes, pred_probas)}
        
        prob_mild = prob_dict.get('Mild', 0.0) * 100
        prob_mod = prob_dict.get('Moderate', 0.0) * 100
        prob_sev = prob_dict.get('Severe', 0.0) * 100
        confianza_ganadora = max(prob_mild, prob_mod, prob_sev)
        
        # Renderizado de Tarjeta Principal sobria y fiel a las clases Mild / Moderate / Severe
        if pred_class == 'Severe':
            st.markdown(f"""
            <div class="triage-card triage-severe">
                <h2 style="margin:0; color:#b91c1c; font-size:1.6rem;">SEVERO (Severe)</h2>
                <p style="margin:6px 0 2px 0; font-size:1.05rem; font-weight:600; color:#991b1b;">
                    Atención Médica Inmediata / Box de Vitales
                </p>
                <p style="margin:0; font-size:0.95rem; color:#7f1d1d;">
                    Certeza del modelo: <b>{confianza_ganadora:.1f}%</b> | Alto riesgo de descompensación ventilatoria o hemodinámica.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.error("Conducta clínica sugerida: Monitorización continua de signos vitales, oxigenoterapia urgente, acceso venoso y aviso inmediato al médico tratante.")
            
        elif pred_class == 'Moderate':
            st.markdown(f"""
            <div class="triage-card triage-moderate">
                <h2 style="margin:0; color:#b45309; font-size:1.6rem;">MODERADO (Moderate)</h2>
                <p style="margin:6px 0 2px 0; font-size:1.05rem; font-weight:600; color:#92400e;">
                    Atención Prioritaria (< 60 minutos)
                </p>
                <p style="margin:0; font-size:0.95rem; color:#78350f;">
                    Certeza del modelo: <b>{confianza_ganadora:.1f}%</b> | Cuadro agudo febril/sistémico sin compromiso vital inminente.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Conducta clínica sugerida: Ubicación en área de observación, medidas antitérmicas/hidratación y reevaluación reglada en 30-45 minutos.")
            
        else: # Mild
            st.markdown(f"""
            <div class="triage-card triage-mild">
                <h2 style="margin:0; color:#15803d; font-size:1.6rem;">LEVE (Mild)</h2>
                <p style="margin:6px 0 2px 0; font-size:1.05rem; font-weight:600; color:#166534;">
                    Atención Estándar Ambulatoria
                </p>
                <p style="margin:0; font-size:0.95rem; color:#14532d;">
                    Certeza del modelo: <b>{confianza_ganadora:.1f}%</b> | Constantes fisiológicas preservadas. Puede aguardar en sala de espera.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.success("Conducta clínica sugerida: Asignar turno general. Pautas de alarma ante aparición de dificultad respiratoria o deterioro súbito.")

        # Explicación contextual si hay discrepancia síntoma vs signo
        if "Fiebre" in selected_symptoms_es and temp < 38.0:
            st.info(f"Nota clínica: El paciente refiere sensación febril, pero la temperatura medida ({temp:.1f}°C) se encuentra bajo el umbral clínico de fiebre (≥ 38.0°C), catalogándose como leve (Mild).")

        # Distribución de Probabilidades
        st.markdown("##### Probabilidades por Clase del Modelo:")
        c_pb1, c_pb2, c_pb3 = st.columns(3)
        c_pb1.metric("Mild (Leve)", f"{prob_mild:.1f}%")
        c_pb2.metric("Moderate (Moderado)", f"{prob_mod:.1f}%")
        c_pb3.metric("Severe (Severo)", f"{prob_sev:.1f}%")

        # Barra de progreso que refleja la confianza de la clase ganadora
        st.progress(confianza_ganadora / 100.0, text=f"Certeza de la predicción [{pred_class}]: {confianza_ganadora:.1f}%")

        # Banderas Clínicas de Riesgo Detectadas
        st.markdown("##### Banderas de Riesgo Fisiológico:")
        alertas = []
        if sat < 95:
            alertas.append(('badge-danger', f'Hipoxemia (SatO₂ {sat}%)'))
        if temp >= 38.0:
            alertas.append(('badge-warning', f'Síndrome Febril ({temp:.1f}°C)'))
        if hr > 100:
            alertas.append(('badge-warning', f'Taquicardia ({hr} lpm)'))
        if shock_idx > 0.9:
            alertas.append(('badge-danger', f'Índice de Shock Elevado ({shock_idx})'))
        if age >= 65:
            alertas.append(('badge-info', f'Paciente Adulto Mayor ({age} años)'))
        if sys_bp >= 160 or dia_bp >= 100:
            alertas.append(('badge-danger', f'Hipertensión ({sys_bp}/{dia_bp})'))
            
        if alertas:
            badges_html = " ".join([f'<span class="metric-badge {cls}">{txt}</span>' for cls, txt in alertas])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.markdown('<span class="metric-badge badge-success">Sin banderas fisiológicas de alarma</span>', unsafe_allow_html=True)

        if evaluar_btn:
            st.session_state.historial_triaje.insert(0, {
                'Hora': pd.Timestamp.now().strftime("%H:%M:%S"),
                'Edad': age,
                'Sexo': gender,
                'SatO2 (%)': sat,
                'Temp (°C)': temp,
                'FC (lpm)': hr,
                'PA (mmHg)': f"{sys_bp}/{dia_bp}",
                'Síntomas': ", ".join(selected_symptoms_es) if selected_symptoms_es else "Ninguno",
                'Triaje': f"{pred_class}",
                'Confianza': f"{confianza_ganadora:.1f}%"
            })
            st.toast(f"Paciente registrado: {pred_class} ({confianza_ganadora:.1f}%)")

    # HISTORIAL DE TRIAJES EN LA SESIÓN
    st.divider()
    st.subheader("Registro de Pacientes Evaluados en la Sesión")
    if len(st.session_state.historial_triaje) > 0:
        df_hist = pd.DataFrame(st.session_state.historial_triaje)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Registro (CSV)",
            data=csv_data,
            file_name="registro_triajes_urgencias.csv",
            mime="text/csv"
        )
    else:
        st.info("Presione 'Registrar Paciente en Bitácora de Triaje' para archivar pacientes evaluados en esta sesión.")

# PESTAÑA 2: ANÁLISIS ESTADÍSTICO, AUDITORÍA DE OUTLIERS Y MODELOS
with tab_analitica:
    st.header("Análisis Bioestadístico, Auditoría de Outliers y Evaluación de Modelos")
    st.markdown("""
    Esta sección consolida el sustento analítico del sistema, la investigación de valores atípicos y las métricas de validación del modelo.
    """)
    
    fig_dir = os.path.join(BASE_DIR, 'reports', 'figures')
    
    # 1. EDA y Distribución
    st.subheader("1. Distribución de Severidad y Sintomatología")
    f1_path = os.path.join(fig_dir, 'fig1_distribucion_severidad_sintomas.png')
    if os.path.exists(f1_path):
        st.image(f1_path, caption="Figura 1: Distribución del nivel de gravedad (A) y prevalencia de síntomas por estrato clínico (B).", use_container_width=True)
    st.markdown(r"""
    - **Fiebre:** Presente en el **100% de los pacientes Moderados** ($\chi^2 = 548.70, p < 10^{-119}$).
    - **Tos persistente:** Presente en el **91.8% de los pacientes Severos** ($\chi^2 = 556.48, p < 10^{-120}$).
    """)
    
    st.divider()
    
    # 2. Auditoría de Outliers
    st.subheader("2. Auditoría de Outliers: Univariados vs. Anomalías Multivariadas")
    f2_path = os.path.join(fig_dir, 'fig2_distribuciones_constantes_outliers.png')
    if os.path.exists(f2_path):
        st.image(f2_path, caption="Figura 2: Distribuciones continuas y límites de Tukey (curtosis ~ -1.20).", use_container_width=True)
        
    st.markdown(r"""
    > **¿Por qué no hay outliers univariados por método IQR?**  
    > Las variables numéricas presentan una curtosis en exceso de aproximadamente **-1.20**, característica exacta de una **distribución continua uniforme $U(a, b)$**. En una distribución uniforme acotada, los límites de Tukey $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ quedan matemáticamente por fuera de los extremos de la distribución ($< a$ y $> b$), por lo que ningún dato univariado supera dichos umbrales.
    """)
    
    f3_path = os.path.join(fig_dir, 'fig3_anomalias_fisiologicas_multivariadas.png')
    if os.path.exists(f3_path):
        st.image(f3_path, caption="Figura 3: Detección de anomalías biológicas: inversión tensional PAS <= PAD (A) y perfil de shock vs SatO2 (B).", use_container_width=True)
        
    st.markdown(r"""
    > **Anomalías Biológicas Reales Detectadas:**  
    > Aunque univariadamente no hay outliers estadísticos, el análisis cruzado reveló **156 casos (7.8%) con PAD $\ge$ PAS**, una inconsistencia biológica atribuible a la generación sintética independiente de variables. Además, con **Isolation Forest** se detectaron 60 casos de shock descompensado extremo.
    """)
    
    st.divider()
    
    # 3. Benchmark y Matriz de Confusión
    st.subheader("3. Benchmark Comparativo y Rendimiento en Conjunto de Prueba ($N=400$)")
    c_m1, c_m2 = st.columns([1.2, 1.0])
    
    with c_m1:
        f4_path = os.path.join(fig_dir, 'fig4_benchmark_comparativa_modelos.png')
        if os.path.exists(f4_path):
            st.image(f4_path, caption="Figura 4: Validación Cruzada Estratificada 5-Fold entre los 6 algoritmos.", use_container_width=True)
            
    with c_m2:
        f5_path = os.path.join(fig_dir, 'fig5_matriz_confusion_test.png')
        if os.path.exists(f5_path):
            st.image(f5_path, caption="Figura 5: Matriz de confusión en el conjunto de prueba independiente.", use_container_width=True)
            
    st.markdown("""
    - **Sensibilidad en Casos Severos:** **100.0%** (76 de 76 casos críticos detectados correctamente, cero falsos negativos).
    - **Macro F1-Score:** **100.0%** en conjunto de test.
    """)
    
    st.divider()
    
    # 4. Importancia de Características
    st.subheader("4. Importancia de Características (Explicabilidad Clínica)")
    f6_path = os.path.join(fig_dir, 'fig6_importancia_caracteristicas.png')
    if os.path.exists(f6_path):
        st.image(f6_path, caption="Figura 6: Ranking de importancia predictiva de Gini (Random Forest).", use_container_width=True)
        
    st.markdown("""
    Las variables clínicas generadas por ingeniería de características (`flag_hypoxia`, `flag_fever`, `flag_tachycardia`, `shock_index` y `map_pressure`) explican más del **27% del poder predictivo del modelo**, demostrando el impacto positivo de modelar conocimiento del dominio de urgencias.
    """)
