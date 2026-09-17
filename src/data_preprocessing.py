"""
Módulo de Preprocesamiento e Ingeniería de Características Clínicas
Proyecto: Triaje Predictivo en Urgencias
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin

LISTA_SINTOMAS_CANONICOS = [
    'Body ache',
    'Cough',
    'Fatigue',
    'Fever',
    'Headache',
    'Runny nose',
    'Shortness of breath',
    'Sore throat'
]

# Mapeo a español para presentación clínica e interfaz
TRADUCCION_SINTOMAS = {
    'Body ache': 'Dolor corporal',
    'Cough': 'Tos persistente',
    'Fatigue': 'Fatiga / Astenia',
    'Fever': 'Fiebre',
    'Headache': 'Cefalea (dolor de cabeza)',
    'Runny nose': 'Congestión / Rinorrea',
    'Shortness of breath': 'Dificultad respiratoria (Disnea)',
    'Sore throat': 'Dolor de garganta'
}

TRADUCCION_INVERSA_SINTOMAS = {v: k for k, v in TRADUCCION_SINTOMAS.items()}

class ClinicalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para extraer variables sintomáticas y calcular
    índices hemodinámicos y fisiológicos de triaje médico.
    """
    def __init__(self, canonical_symptoms: List[str] = None):
        self.canonical_symptoms = canonical_symptoms or LISTA_SINTOMAS_CANONICOS

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Desglose de Presión Arterial si viene en formato string
        if 'Blood_Pressure_mmHg' in df.columns:
            if df['Blood_Pressure_mmHg'].dtype == object or isinstance(df['Blood_Pressure_mmHg'].iloc[0], str):
                bp_split = df['Blood_Pressure_mmHg'].astype(str).str.split('/', expand=True)
                df['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120).astype(float)
                df['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80).astype(float)
        
        # Asegurar columnas numéricas
        sys = df['BP_Systolic'].astype(float)
        dia = df['BP_Diastolic'].astype(float)
        hr = df['Heart_Rate_bpm'].astype(float)
        temp = df['Body_Temperature_C'].astype(float)
        sat = df['Oxygen_Saturation_%'].astype(float)
        age = df['Age'].astype(float)
        
        # 2. Codificación binaria de síntomas
        for sym in self.canonical_symptoms:
            sym_clean_name = f"sym_{sym.lower().replace(' ', '_')}"
            
            # Soporta tanto formato multi-columna (Symptom_1, Symptom_2, Symptom_3)
            # como lista de síntomas provista por la interfaz
            if 'Symptom_1' in df.columns and 'Symptom_2' in df.columns and 'Symptom_3' in df.columns:
                df[sym_clean_name] = (
                    (df['Symptom_1'] == sym) |
                    (df['Symptom_2'] == sym) |
                    (df['Symptom_3'] == sym)
                ).astype(int)
            elif 'Symptoms_List' in df.columns:
                df[sym_clean_name] = df['Symptoms_List'].apply(
                    lambda s_list: 1 if isinstance(s_list, (list, set, tuple)) and sym in s_list else 0
                ).astype(int)
            else:
                if sym_clean_name not in df.columns:
                    df[sym_clean_name] = 0

        # Conteo total de síntomas referidos
        sym_cols = [f"sym_{sym.lower().replace(' ', '_')}" for sym in self.canonical_symptoms]
        df['symptom_count'] = df[sym_cols].sum(axis=1)

        # 3. Índices Clínicos y Hemodinámicos
        # Índice de Shock (FC / Presión Sistólica) - Normal: 0.5 - 0.7; > 0.9 indica shock o inestabilidad
        df['shock_index'] = (hr / sys.replace(0, 120)).round(3)
        
        # Presión Arterial Media (PAM / MAP): Diastólica + (Sistólica - Diastólica) / 3
        df['map_pressure'] = (dia + (sys - dia) / 3.0).round(2)
        
        # Indicadores clínicos discretos (Alertas de signos vitales)
        df['flag_hypoxia'] = (sat < 95.0).astype(int)       # Hipoxemia
        df['flag_fever'] = (temp >= 38.0).astype(int)         # Síndrome febril
        df['flag_tachycardia'] = (hr > 100.0).astype(int)     # Taquicardia
        df['flag_elderly'] = (age >= 65.0).astype(int)        # Paciente geriátrico vulnerable

        # 4. Género
        if 'Gender' in df.columns:
            df['is_male'] = (df['Gender'].astype(str).str.lower().isin(['male', 'masculino', 'm'])).astype(int)
        else:
            df['is_male'] = 0

        # Seleccionar features finales para el modelo
        feature_columns = [
            'Age',
            'is_male',
            'Heart_Rate_bpm',
            'Body_Temperature_C',
            'BP_Systolic',
            'BP_Diastolic',
            'Oxygen_Saturation_%',
            'shock_index',
            'map_pressure',
            'flag_hypoxia',
            'flag_fever',
            'flag_tachycardia',
            'flag_elderly',
            'symptom_count'
        ] + sym_cols
        
        return df[feature_columns]

def preparar_datos_entrenamiento(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Carga el dataset fuente y separa características (X) y variable objetivo (y: Severity).
    Excluye estrictamente Diagnosis y Treatment_Plan para evitar fuga de datos (data leakage).
    """
    df = pd.read_csv(csv_path)
    
    # Target
    y = df['Severity'].copy()
    
    # Features crudas
    X_raw = df.drop(columns=['Patient_ID', 'Diagnosis', 'Treatment_Plan', 'Severity'])
    
    extractor = ClinicalFeatureExtractor()
    X_engineered = extractor.fit_transform(X_raw)
    feature_names = X_engineered.columns.tolist()
    
    return X_engineered, y, feature_names
