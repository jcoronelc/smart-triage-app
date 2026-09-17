"""
Módulo de Análisis Exploratorio de Datos (EDA) y Pruebas Estadísticas
Proyecto: Triaje Predictivo en Urgencias
Dataset: disease_diagnosis.csv
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calcula la V de Cramér para medir la fuerza de asociación entre variables categóricas."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    min_dim = min((kcorr - 1), (rcorr - 1))
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / min_dim))

def ejecutar_eda(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    # 1. Separación de Presión Arterial
    bp_split = df['Blood_Pressure_mmHg'].str.split('/', expand=True).astype(int)
    df['BP_Systolic'] = bp_split[0]
    df['BP_Diastolic'] = bp_split[1]
    
    # 2. Resumen General
    n_rows, n_cols = df.shape
    missing = df.isnull().sum().to_dict()
    severity_counts = df['Severity'].value_counts().to_dict()
    severity_pct = (df['Severity'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # 3. Análisis de Síntomas
    sintomas_unicos = sorted(list(set(
        df['Symptom_1'].dropna().unique().tolist() +
        df['Symptom_2'].dropna().unique().tolist() +
        df['Symptom_3'].dropna().unique().tolist()
    )))
    
    symptom_presence = pd.DataFrame(index=df.index)
    for sym in sintomas_unicos:
        symptom_presence[sym] = (
            (df['Symptom_1'] == sym) | 
            (df['Symptom_2'] == sym) | 
            (df['Symptom_3'] == sym)
        ).astype(int)
    
    symptom_tests = {}
    for sym in sintomas_unicos:
        crosstab = pd.crosstab(symptom_presence[sym], df['Severity'])
        chi2, p_val, dof, _ = stats.chi2_contingency(crosstab)
        cv = cramers_v(crosstab)
        prevalencia_por_severidad = df.groupby('Severity').apply(
            lambda g, s=sym: float(((g['Symptom_1'] == s) | (g['Symptom_2'] == s) | (g['Symptom_3'] == s)).mean() * 100)
        ).to_dict()
        
        symptom_tests[sym] = {
            'chi2': round(chi2, 3),
            'p_value': float(f"{p_val:.3e}"),
            'cramers_v': round(cv, 4),
            'prevalencia_por_severidad_pct': prevalencia_por_severidad
        }
        
    # 4. Análisis de Signos Vitales Numéricos por Severidad
    vitals = ['Age', 'Heart_Rate_bpm', 'Body_Temperature_C', 'BP_Systolic', 'BP_Diastolic', 'Oxygen_Saturation_%']
    vitals_stats = {}
    
    for v in vitals:
        # Test Kruskal-Wallis (robusto a no-normalidad y asimetrías)
        mild_vals = df[df['Severity'] == 'Mild'][v].values
        mod_vals = df[df['Severity'] == 'Moderate'][v].values
        sev_vals = df[df['Severity'] == 'Severe'][v].values
        
        kw_stat, kw_p = stats.kruskal(mild_vals, mod_vals, sev_vals)
        
        vitals_stats[v] = {
            'global_mean': round(float(df[v].mean()), 2),
            'global_std': round(float(df[v].std()), 2),
            'by_severity': {
                'Mild': {'mean': round(float(mild_vals.mean()), 2), 'median': round(float(np.median(mild_vals)), 2), 'std': round(float(mild_vals.std()), 2)},
                'Moderate': {'mean': round(float(mod_vals.mean()), 2), 'median': round(float(np.median(mod_vals)), 2), 'std': round(float(mod_vals.std()), 2)},
                'Severe': {'mean': round(float(sev_vals.mean()), 2), 'median': round(float(np.median(sev_vals)), 2), 'std': round(float(sev_vals.std()), 2)},
            },
            'kruskal_wallis_h': round(float(kw_stat), 3),
            'p_value': float(f"{kw_p:.3e}")
        }
        
    # 5. Género vs Severidad
    gender_ct = pd.crosstab(df['Gender'], df['Severity'])
    g_chi2, g_pval, _, _ = stats.chi2_contingency(gender_ct)
    g_cv = cramers_v(gender_ct)
    
    resultados = {
        'total_registros': n_rows,
        'total_columnas': n_cols,
        'distribucion_severidad': {
            'conteo': severity_counts,
            'porcentaje': severity_pct
        },
        'pruebas_sintomas': symptom_tests,
        'pruebas_signos_vitales': vitals_stats,
        'genero_vs_severidad': {
            'chi2': round(g_chi2, 3),
            'p_value': float(f"{g_pval:.3e}"),
            'cramers_v': round(g_cv, 4)
        }
    }
    
    return resultados

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'disease_diagnosis.csv')
    res = ejecutar_eda(data_path)
    
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    report_json_path = os.path.join(output_dir, 'eda_summary.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        
    print(f"EDA completado exitosamente. Resumen guardado en: {report_json_path}")
    print("\n--- DISTRIBUCIÓN DE SEVERIDAD ---")
    for k, v in res['distribucion_severidad']['porcentaje'].items():
        print(f"  {k}: {v}% ({res['distribucion_severidad']['conteo'][k]} pacientes)")
        
    print("\n--- ASOCIACIÓN DE SÍNTOMAS CON SEVERIDAD (Chi2 y V de Cramér) ---")
    for s, data in res['pruebas_sintomas'].items():
        print(f"  {s:<22} | Chi2={data['chi2']:>7.2f} | p={data['p_value']} | V={data['cramers_v']:>6.4f}")
        
    print("\n--- DIFERENCIA DE SIGNOS VITALES ENTRE SEVERIDADES (Kruskal-Wallis) ---")
    for v, data in res['pruebas_signos_vitales'].items():
        print(f"  {v:<22} | H={data['kruskal_wallis_h']:>7.2f} | p={data['p_value']}")
