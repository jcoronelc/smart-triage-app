"""
Script para generar gráficos de alta resolución para el informe final
Proyecto: Triaje Predictivo en Urgencias
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Cargar datos
csv_path = os.path.join(BASE_DIR, 'data', 'disease_diagnosis.csv')
df = pd.read_csv(csv_path)

bp_split = df['Blood_Pressure_mmHg'].str.split('/', expand=True).astype(int)
df['BP_Systolic'] = bp_split[0]
df['BP_Diastolic'] = bp_split[1]
df['Pulse_Pressure'] = df['BP_Systolic'] - df['BP_Diastolic']
df['Shock_Index'] = df['Heart_Rate_bpm'] / df['BP_Systolic']

COLORES_SEVERIDAD = {'Mild': '#2ecc71', 'Moderate': '#f39c12', 'Severe': '#e74c3c'}
PALETA_SEVERIDAD = ['#2ecc71', '#f39c12', '#e74c3c']

# ==============================================================================
# FIGURA 1: Distribución de Severidad y Prevalencia de Síntomas
# ==============================================================================
print("Generando Figura 1: Distribución de Severidad y Prevalencia de Síntomas...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Gráfico 1A: Conteo y porcentaje de Severidad
order_sev = ['Mild', 'Moderate', 'Severe']
counts = df['Severity'].value_counts()[order_sev]
pcts = (counts / len(df) * 100).round(1)

bars = ax1.bar(order_sev, counts, color=PALETA_SEVERIDAD, width=0.55, edgecolor='black', alpha=0.85)
ax1.set_title('A. Distribución de Severidad de Pacientes en Admisión (N=2000)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Nivel de Triaje (Severidad)', fontsize=11, labelpad=8)
ax1.set_ylabel('Frecuencia de Pacientes', fontsize=11, labelpad=8)
ax1.set_ylim(0, 1550)

for bar, count, pct in zip(bars, counts, pcts):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 25, f"{count:,}\n({pct}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Gráfico 1B: Prevalencia de Síntomas por Severidad
sintomas = ['Fever', 'Cough', 'Shortness of breath', 'Fatigue', 'Body ache', 'Sore throat', 'Runny nose', 'Headache']
trad_sint = {
    'Fever': 'Fiebre', 'Cough': 'Tos', 'Shortness of breath': 'Disnea',
    'Fatigue': 'Fatiga', 'Body ache': 'Mialgias', 'Sore throat': 'Odinofagia',
    'Runny nose': 'Rinorrea', 'Headache': 'Cefalea'
}

data_sint = []
for s in sintomas:
    has_s = ((df['Symptom_1'] == s) | (df['Symptom_2'] == s) | (df['Symptom_3'] == s)).astype(int)
    for sev in order_sev:
        sub_s = has_s[df['Severity'] == sev]
        data_sint.append({'Síntoma': trad_sint[s], 'Severidad': sev, 'Prevalencia (%)': sub_s.mean() * 100})

df_prev = pd.DataFrame(data_sint)
sns.barplot(data=df_prev, y='Síntoma', x='Prevalencia (%)', hue='Severidad', palette=PALETA_SEVERIDAD, ax=ax2, edgecolor='black', alpha=0.85)
ax2.set_title('B. Prevalencia de Síntomas según Nivel de Severidad (%)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Prevalencia en Estrato (%)', fontsize=11, labelpad=8)
ax2.set_ylabel('')
ax2.set_xlim(0, 110)
ax2.legend(title='Nivel de Triaje', loc='lower right', frameon=True)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'fig1_distribucion_severidad_sintomas.png')
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURA 2: Distribuciones Univariadas y Explicación Matemática de Outliers
# ==============================================================================
print("Generando Figura 2: Distribuciones Univariadas y Límites de Tukey...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
vitals_info = [
    ('Age', 'Edad (años)', 'años'),
    ('Heart_Rate_bpm', 'Frecuencia Cardíaca (lpm)', 'lpm'),
    ('Body_Temperature_C', 'Temperatura Corporal (°C)', '°C'),
    ('BP_Systolic', 'Presión Sistólica (mmHg)', 'mmHg'),
    ('BP_Diastolic', 'Presión Diastólica (mmHg)', 'mmHg'),
    ('Oxygen_Saturation_%', 'Saturación O₂ (%)', '%')
]

for idx, (col, title, unit) in enumerate(vitals_info):
    ax = axes[idx // 3, idx % 3]
    vals = df[col]
    
    # Histograma con densidad
    sns.histplot(vals, bins=25, kde=True, ax=ax, color='#3498db', edgecolor='black', alpha=0.6, stat="density")
    
    # Límites de Tukey IQR
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    lower_tukey = q1 - 1.5 * iqr
    upper_tukey = q3 + 1.5 * iqr
    
    ax.axvline(q1, color='#2c3e50', linestyle=':', linewidth=1.5, label=f'Q1={q1:.1f}')
    ax.axvline(q3, color='#2c3e50', linestyle=':', linewidth=1.5, label=f'Q3={q3:.1f}')
    ax.axvline(vals.median(), color='#e74c3c', linestyle='-', linewidth=2, label=f'Mediana={vals.median():.1f}')
    
    kurt = vals.kurtosis()
    ax.set_title(f"{title}\nCurtosis = {kurt:.2f} (Uniforme ~ -1.20)", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"{title}")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle("Distribuciones Fisiológicas Univariadas: Explicación de Ausencia de Outliers Univariados por Soporte Compacto Uniforme", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'fig2_distribuciones_constantes_outliers.png')
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURA 3: Anomalías Multivariadas e Inconsistencias Fisiológicas
# ==============================================================================
print("Generando Figura 3: Anomalías Fisiológicas e Inconsistencias Multivariadas...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# 3A: Presión Sistólica vs Diastólica (Inconsistencia PAS <= PAD)
anomalia_presion = df['Pulse_Pressure'] <= 0
sns.scatterplot(
    data=df,
    x='BP_Systolic',
    y='BP_Diastolic',
    hue=anomalia_presion.map({True: 'Inconsistente (PAD >= PAS)', False: 'Fisiológica (PAS > PAD)'}),
    palette={'Inconsistente (PAD >= PAS)': '#e74c3c', 'Fisiológica (PAS > PAD)': '#2980b9'},
    alpha=0.7,
    s=45,
    ax=ax1
)
ax1.plot([80, 180], [80, 180], 'r--', linewidth=1.5, label='Línea de Identidad (PAS = PAD)')
ax1.set_title(f'A. Dispersión Tensional (156 casos con PAD ≥ PAS)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Presión Sistólica (mmHg)', fontsize=11)
ax1.set_ylabel('Presión Diastólica (mmHg)', fontsize=11)
ax1.legend(loc='upper left', frameon=True)

# 3B: Saturación O2 vs Índice de Shock coloreado por Severidad
sns.scatterplot(
    data=df,
    x='Oxygen_Saturation_%',
    y='Shock_Index',
    hue='Severity',
    hue_order=['Mild', 'Moderate', 'Severe'],
    palette=COLORES_SEVERIDAD,
    alpha=0.75,
    s=55,
    ax=ax2
)
ax2.axhline(0.9, color='#c0392b', linestyle='--', linewidth=1.5, label='Umbral Alerta Shock (>0.9)')
ax2.axvline(95.0, color='#d35400', linestyle='--', linewidth=1.5, label='Umbral Hipoxemia (<95%)')
ax2.set_title('B. Perfil Hemodinámico: Saturación O₂ vs Índice de Shock', fontsize=12, fontweight='bold')
ax2.set_xlabel('Saturación de Oxígeno (%)', fontsize=11)
ax2.set_ylabel('Índice de Shock (FC / PAS)', fontsize=11)
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, 'fig3_anomalias_fisiologicas_multivariadas.png')
plt.savefig(fig3_path, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURA 4: Benchmark de Modelos (Validación Cruzada 5-Fold)
# ==============================================================================
print("Generando Figura 4: Benchmark Comparativo de Modelos...")
metadata_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
with open(metadata_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

df_bench = pd.DataFrame(meta['benchmark_cv_5fold'])
df_bench['Macro F1-Score (%)'] = df_bench['Macro F1-Score'] * 100
df_bench['Accuracy Media (%)'] = df_bench['Accuracy Media'] * 100
df_bench['Balanced Accuracy (%)'] = df_bench['Balanced Accuracy'] * 100

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
x = np.arange(len(df_bench))
width = 0.26

r1 = ax.bar(x - width, df_bench['Macro F1-Score (%)'], width, label='Macro F1-Score (%)', color='#2980b9', edgecolor='black', alpha=0.9)
r2 = ax.bar(x, df_bench['Accuracy Media (%)'], width, label='Exactitud (Accuracy %)', color='#27ae60', edgecolor='black', alpha=0.9)
r3 = ax.bar(x + width, df_bench['Balanced Accuracy (%)'], width, label='Balanced Accuracy (%)', color='#f39c12', edgecolor='black', alpha=0.9)

ax.set_ylabel('Puntuación de Rendimiento (%)', fontsize=11)
ax.set_title('Comparativa de Modelos de Machine Learning (Validación Cruzada Estratificada 5-Fold)', fontsize=13, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' Classifier', '').replace(' (Multinomial)', '') for m in df_bench['Modelo']], rotation=15, ha='right', fontsize=10)
ax.set_ylim(80, 105)
ax.legend(loc='lower right', frameon=True)

for bars in [r1, r2, r3]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h + 0.5, f"{h:.1f}", ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, 'fig4_benchmark_comparativa_modelos.png')
plt.savefig(fig4_path, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURA 5: Matriz de Confusión en Test Set (N=400)
# ==============================================================================
print("Generando Figura 5: Matriz de Confusión...")
cm = np.array(meta['test_metrics']['confusion_matrix']['matrix'])
labels = meta['test_metrics']['confusion_matrix']['labels']

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=[f"Predicho {l}" for l in labels],
            yticklabels=[f"Real {l}" for l in labels],
            annot_kws={'size': 14, 'fontweight': 'bold'})

ax.set_title('Matriz de Confusión en Conjunto de Prueba Independiente (N=400)\nRecall en Casos Severos = 100.0% (Cero Falsos Negativos)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Clase Predicha por el Pipeline de Triaje', fontsize=11, labelpad=8)
ax.set_ylabel('Clase Real del Paciente', fontsize=11, labelpad=8)

plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, 'fig5_matriz_confusion_test.png')
plt.savefig(fig5_path, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURA 6: Importancia de Características (Top 12)
# ==============================================================================
print("Generando Figura 6: Importancia de Características...")
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'triaje_model.joblib'))
rf = pipeline.named_steps['classifier']
feat_names = meta['feature_names']
importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=True)

# Traducir nombres para el gráfico
trad_features = {
    'sym_fever': 'Síntoma: Fiebre',
    'sym_cough': 'Síntoma: Tos persistente',
    'Body_Temperature_C': 'Temperatura Corporal (°C)',
    'flag_hypoxia': 'Alerta: Hipoxemia (SatO2 < 95%)',
    'Oxygen_Saturation_%': 'Saturación de Oxígeno (%)',
    'flag_fever': 'Alerta: Síndrome Febril (≥38°C)',
    'Heart_Rate_bpm': 'Frecuencia Cardíaca (lpm)',
    'flag_tachycardia': 'Alerta: Taquicardia (>100 lpm)',
    'shock_index': 'Índice de Shock (FC / PAS)',
    'map_pressure': 'Presión Arterial Media (PAM)',
    'sym_shortness_of_breath': 'Síntoma: Disnea',
    'sym_fatigue': 'Síntoma: Fatiga / Astenia',
    'sym_body_ache': 'Síntoma: Mialgias',
    'BP_Systolic': 'Presión Sistólica (mmHg)',
    'BP_Diastolic': 'Presión Diastólica (mmHg)',
    'Age': 'Edad del Paciente',
    'symptom_count': 'Conteo Total de Síntomas'
}
imp_top = importances.tail(12)
imp_top_named = [trad_features.get(f, f) for f in imp_top.index]

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
colors = ['#34495e' if 'Alerta' in n or 'Índice' in n or 'Presión Arterial Media' in n else '#2980b9' for n in imp_top_named]

bars = ax.barh(imp_top_named, imp_top.values * 100, color=colors, edgecolor='black', alpha=0.85, height=0.65)
ax.set_title('Top 12 Variables con Mayor Importancia Predictiva (Random Forest)\n(Barras grises indican variables generadas por Ingeniería de Características)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Importancia Relativa de Gini (%)', fontsize=11, labelpad=8)
ax.set_xlim(0, max(imp_top.values * 100) * 1.15)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.4, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}%", ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, 'fig6_importancia_caracteristicas.png')
plt.savefig(fig6_path, bbox_inches='tight')
plt.close()

print("¡Todas las figuras fueron generadas con éxito en reports/figures/!")
