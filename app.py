"""
Aplicación Web de Triaje Predictivo en Urgencias con Asistente Graph RAG y Ollama
Desarrollada con Streamlit, Scikit-Learn, NetworkX y Ollama
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
from src.medical_knowledge_graph import obtener_grafo_medico
from src.ollama_client import (
    verificar_conexion_ollama,
    stream_chat_ollama,
    construir_prompt_sistema_graph_rag,
    generar_respuesta_fallback_grafo,
    DEFAULT_OLLAMA_URL,
    DEFAULT_MODEL
)

# Configuración de página
st.set_page_config(
    page_title="Triaje Predictivo | Urgencias & Graph RAG",
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
    .graph-evidence-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Carga del Modelo y Grafo en Caché
@st.cache_resource(show_spinner="Cargando pipeline de triaje y grafo ontológico...")
def cargar_recursos():
    model_path = os.path.join(BASE_DIR, 'models', 'triaje_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Archivo de modelo no encontrado en: {model_path}. Ejecute src/train_and_evaluate.py primero.")
        st.stop()
    pipeline_model = joblib.load(model_path)
    grafo = obtener_grafo_medico()
    return pipeline_model, grafo

pipeline, medical_graph = cargar_recursos()

# Inicialización de estado en sesión
if 'historial_triaje' not in st.session_state:
    st.session_state.historial_triaje = []

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {
            'role': 'assistant',
            'content': 'Hola, soy el Asistente Clínico de Triaje con Graph RAG. Puedo responder tus dudas sobre el caso clínico actual, consultar precauciones de la ontología médica y verificar signos de alarma. ¿En qué puedo orientarte?'
        }
    ]

if 'ollama_server_url' not in st.session_state:
    st.session_state.ollama_server_url = DEFAULT_OLLAMA_URL

if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = DEFAULT_MODEL

# Variables de sincronización de widgets
if 'age_key' not in st.session_state: st.session_state['age_key'] = 45
if 'gender_key' not in st.session_state: st.session_state['gender_key'] = "Femenino"
if 'hr_key' not in st.session_state: st.session_state['hr_key'] = 80
if 'temp_key' not in st.session_state: st.session_state['temp_key'] = 36.8
if 'sys_key' not in st.session_state: st.session_state['sys_key'] = 120
if 'dia_key' not in st.session_state: st.session_state['dia_key'] = 80
if 'sat_key' not in st.session_state: st.session_state['sat_key'] = 98
if 'symptoms_key' not in st.session_state: st.session_state['symptoms_key'] = ["Dolor de garganta", "Congestión / Rinorrea"]

# Estado del paciente activo para Graph RAG
if 'ultimo_paciente_evaluado' not in st.session_state:
    st.session_state.ultimo_paciente_evaluado = {
        'Edad': 45,
        'Sexo': 'Femenino',
        'FC': 80,
        'Temp': 36.8,
        'PA': '120/80',
        'SatO2': 98,
        'Sintomas': 'Dolor de garganta, Congestión / Rinorrea',
        'Triaje': 'Mild',
        'Confianza': '95.0%'
    }

# SIDEBAR: Información del Modelo y Disclaimer
with st.sidebar:
    st.title("Triaje Predictivo")
    st.caption("Sistema de Apoyo Clínico con Graph RAG & Ollama")
    
    st.divider()
    st.markdown("### Especificaciones del Sistema")
    st.markdown(f"""
    - **Clasificador:** Random Forest (200 estimadores)
    - **Validación ML:** 5-Fold Stratified CV (100% Macro F1)
    - **Grafo de Conocimiento:** {medical_graph.graph.number_of_nodes()} nodos, {medical_graph.graph.number_of_edges()} aristas
    - **Motor LLM:** Ollama Local / Servidor Remoto
    """)
    
    st.divider()
    st.markdown("### Clasificación de Severidad")
    st.markdown("""
    - **Severe (Severo):** Atención médica inmediata en box de soporte vital / reanimación.
    - **Moderate (Moderado):** Atención prioritaria sugerida en < 60 min.
    - **Mild (Leve):** Manejo ambulatorio estándar / sala de espera.
    """)
    
    st.divider()
    st.info("""
    **Descargo de Responsabilidad:**  
    Prototipo desarrollado con fines académicos. No reemplaza el juicio clínico médico profesional ni la evaluación presencial de enfermería.
    """)

# HEADER PRINCIPAL
st.markdown('<div class="main-header">Sistema de Triaje Predictivo & Asistente Clínico Graph RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clasificación asistida de gravedad clínica y razonamiento conversacional anclado a ontología médica con LLM local/servidor.</div>', unsafe_allow_html=True)

# Pestañas Principales
tab_triaje, tab_chatbot, tab_analitica = st.tabs([
    "Triaje en Vivo",
    "Chatbot Clínico (Graph RAG & Ollama)",
    "Análisis Estadístico y Modelos"
])

# ==============================================================================
# PESTAÑA 1: TRIAJE EN VIVO
# ==============================================================================
with tab_triaje:
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

        # Actualizar paciente activo en sesión
        st.session_state.ultimo_paciente_evaluado = {
            'Edad': age,
            'Sexo': gender,
            'FC': hr,
            'Temp': temp,
            'PA': f"{sys_bp}/{dia_bp}",
            'SatO2': sat,
            'Sintomas': ", ".join(selected_symptoms_es) if selected_symptoms_es else "Ninguno",
            'Triaje': pred_class,
            'Confianza': f"{confianza_ganadora:.1f}%"
        }
        
        # Renderizado de Tarjeta Principal
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

# ==============================================================================
# PESTAÑA 2: CHATBOT CLÍNICO (GRAPH RAG & OLLAMA)
# ==============================================================================
with tab_chatbot:
    st.subheader("Asistente Clínico Inteligente (Graph RAG & Ollama Local / Servidor)")
    st.markdown("""
    Este agente combina **Razonamiento con LLMs (Ollama)** anclado a un **Grafo de Conocimiento Médico (Graph RAG)**. 
    Las respuestas no son inventadas por el modelo de lenguaje: están respaldadas en la ontología estructurada de síntomas, precauciones y terapias.
    """)

    # 1. Panel de Configuración de Conexión a Ollama
    with st.expander("Configuración del Servidor Ollama (Local o Remoto en tu Computador)", expanded=False):
        c_srv1, c_srv2, c_srv3 = st.columns([1.8, 1.2, 0.8])
        with c_srv1:
            ollama_url_input = st.text_input(
                "URL del Servidor Ollama:",
                value=st.session_state.ollama_server_url,
                help="Por defecto 'http://localhost:11434'. Si la app corre en Streamlit Cloud, ingrese aquí la URL pública de su túnel (Ngrok o Cloudflare) apuntando a su computador."
            )
        with c_srv2:
            # Comprobación de modelos disponibles
            is_connected, available_models, status_msg = verificar_conexion_ollama(ollama_url_input)
            model_options = available_models if available_models else ["qwen3.5:9b", "deepseek-r1:8b", "qwen3.8:latest"]
            selected_model = st.selectbox(
                "Modelo LLM en el Servidor:",
                options=model_options,
                index=0 if st.session_state.ollama_model not in model_options else model_options.index(st.session_state.ollama_model)
            )
            st.session_state.ollama_model = selected_model
        with c_srv3:
            st.write("")
            st.write("")
            test_conn_btn = st.button("Probar Conexión", use_container_width=True)

        st.session_state.ollama_server_url = ollama_url_input

        if is_connected:
            st.success(f"Conectado exitosamente al servidor Ollama ({ollama_url_input}). Modelos detectados: {', '.join(available_models)}")
        else:
            st.warning(f"{status_msg} (El chatbot operará en modo de respaldo estructurado con el Grafo de Conocimiento).")

        st.markdown("""
        **¿Cómo conectar esta app desde internet a tu computador como servidor Ollama?**
        1. Asegúrate de tener Ollama corriendo en tu terminal: `ollama serve`
        2. En otra terminal de tu Mac, abre un túnel gratuito hacia el puerto 11434:
           - **Con Cloudflare:** `brew install cloudflared && cloudflared tunnel --url http://localhost:11434`
           - **O con Ngrok:** `ngrok http 11434`
           - **O con LocalTunnel:** `npx localtunnel --port 11434`
        3. Copia la URL pública generada (ej: `https://xxxx.trycloudflare.com`) y pégala en el campo **URL del Servidor Ollama** arriba.
        """)

    # 2. Contexto de Triaje y Evidencia de Graph RAG
    paciente = st.session_state.ultimo_paciente_evaluado
    graph_ctx = medical_graph.retrieve_clinical_context(
        severity=paciente.get('Triaje', 'Mild'),
        symptoms=[s.strip() for s in paciente.get('Sintomas', '').split(',') if s.strip()]
    )

    col_ctx1, col_ctx2 = st.columns([1.1, 1.3])
    with col_ctx1:
        st.markdown(f"""
        <div style="background-color:#f1f5f9; padding:0.9rem; border-radius:8px; border:1px solid #cbd5e1; font-size:0.88rem;">
            <b>Caso Clínico Vinculado al Chat:</b><br>
            • <b>Paciente:</b> {paciente.get('Edad')} años, {paciente.get('Sexo')}<br>
            • <b>Signos:</b> FC {paciente.get('FC')} lpm | Temp {paciente.get('Temp')} °C | PA {paciente.get('PA')} mmHg | SatO₂ {paciente.get('SatO2')}%<br>
            • <b>Síntomas:</b> {paciente.get('Sintomas')}<br>
            • <b>Triaje ML:</b> <b>{paciente.get('Triaje')}</b> (Certeza: {paciente.get('Confianza')})
        </div>
        """, unsafe_allow_html=True)

    with col_ctx2:
        with st.expander("Ver Evidencia Estructurada del Grafo de Conocimiento (Graph RAG)", expanded=False):
            st.markdown(f"**Nivel de Acción Ontológica:** {graph_ctx['accion_triaje']}")
            st.markdown(f"**Precauciones Validadas:** {', '.join(graph_ctx['precauciones_validadas'])}")
            st.markdown(f"**Fármacos Asociados en el Grafo:** {', '.join(graph_ctx['farmacos_ontologia'])}")
            st.markdown(f"**Dietas y Cuidados:** {', '.join(graph_ctx['dietas_ontologia'])}")
            st.caption(f"Grafo de referencia: {graph_ctx['resumen_grafo']['total_nodos']} nodos y {graph_ctx['resumen_grafo']['total_aristas']} aristas ontológicas.")
            
            st.markdown("---")
            st.markdown("<b>Exportar Grafo de Conocimiento a Disco:</b>", unsafe_allow_html=True)
            cg1, cg2 = st.columns(2)
            graphml_file = os.path.join(BASE_DIR, 'data', 'knowledge_graph', 'medical_knowledge_graph.graphml')
            json_file = os.path.join(BASE_DIR, 'data', 'knowledge_graph', 'medical_knowledge_graph.json')
            
            if os.path.exists(graphml_file):
                with open(graphml_file, 'rb') as f:
                    cg1.download_button(
                        label="Descargar GraphML (Gephi)",
                        data=f.read(),
                        file_name="medical_knowledge_graph.graphml",
                        mime="application/xml",
                        use_container_width=True
                    )
            if os.path.exists(json_file):
                with open(json_file, 'rb') as f:
                    cg2.download_button(
                        label="Descargar JSON (D3.js)",
                        data=f.read(),
                        file_name="medical_knowledge_graph.json",
                        mime="application/json",
                        use_container_width=True
                    )

    st.divider()

    # 3. Sugerencias Rápidas de Preguntas
    col_q1, col_q2, col_q3 = st.columns(3)
    pregunta_sugerida = None
    with col_q1:
        if st.button("¿Cuáles son los signos de alarma para urgencias?", use_container_width=True):
            pregunta_sugerida = "¿Cuáles son los signos de alarma que requieren atención médica urgente para este paciente?"
    with col_q2:
        if st.button("¿Qué precauciones indica el grafo médico?", use_container_width=True):
            pregunta_sugerida = "¿Qué precauciones y cuidados específicos recomienda la ontología médica según el triaje?"
    with col_q3:
        if st.button("¿Qué manejo farmacológico y dieta sugiere el grafo?", use_container_width=True):
            pregunta_sugerida = "¿Qué medicamentos y pautas de hidratación están registrados en el grafo para esta severidad?"

    # 4. Historial del Chat
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Entrada del Chat
    chat_input = st.chat_input("Escribe una consulta sobre el paciente, su severidad o precauciones...")
    prompt_usuario = pregunta_sugerida if pregunta_sugerida else chat_input

    if prompt_usuario:
        # Registrar y mostrar mensaje de usuario
        st.session_state.chat_messages.append({'role': 'user', 'content': prompt_usuario})
        with st.chat_message('user'):
            st.markdown(prompt_usuario)

        # Construir System Prompt con Graph RAG
        system_prompt = construir_prompt_sistema_graph_rag(graph_ctx, paciente)

        with st.chat_message('assistant'):
            # Si el servidor Ollama está conectado, generar con streaming
            if is_connected:
                historial_para_ollama = [
                    m for m in st.session_state.chat_messages if m['role'] in ['user', 'assistant']
                ]
                stream_generator = stream_chat_ollama(
                    mensajes=historial_para_ollama,
                    model=st.session_state.ollama_model,
                    base_url=st.session_state.ollama_server_url,
                    system_prompt=system_prompt
                )
                respuesta_completa = st.write_stream(stream_generator)
            else:
                # Modo de respaldo directo desde el grafo de conocimiento
                respuesta_completa = generar_respuesta_fallback_grafo(graph_ctx, paciente, prompt_usuario)
                st.markdown(respuesta_completa)

        st.session_state.chat_messages.append({'role': 'assistant', 'content': respuesta_completa})

    if len(st.session_state.chat_messages) > 2:
        if st.button("Limpiar Conversación"):
            st.session_state.chat_messages = [st.session_state.chat_messages[0]]
            st.rerun()

# ==============================================================================
# PESTAÑA 3: ANÁLISIS ESTADÍSTICO, AUDITORÍA DE OUTLIERS Y MODELOS
# ==============================================================================
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
