"""
Módulo de Entrenamiento, Comparativa de Modelos y Exportación del Pipeline
Proyecto: Triaje Predictivo en Urgencias
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Imports locales
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import ClinicalFeatureExtractor, preparar_datos_entrenamiento

def benchmark_modelos(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Ejecuta validación cruzada estratificada (5 folds) sobre un conjunto diverso
    de clasificadores para determinar el algoritmo óptimo en el triaje de urgencias.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Mapeo de severidad a orden clínico estándar
    classes = ['Mild', 'Moderate', 'Severe']
    
    modelos = {
        'Logistic Regression (Multinomial)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        'K-Nearest Neighbors (k-NN)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=7))
        ]),
        'Support Vector Machine (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=42, class_weight='balanced'))
        ]),
        'Multilayer Perceptron (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42))
        ]),
        'Random Forest Classifier': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'))
        ]),
        'XGBoost Classifier': Pipeline([
            ('clf', XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                eval_metric='mlogloss'
            ))
        ])
    }
    
    # Codificar clases para XGBoost si es necesario
    y_encoded = y.map({'Mild': 0, 'Moderate': 1, 'Severe': 2})
    
    resultados = []
    
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }
    
    for nombre, pipe in modelos.items():
        y_target = y_encoded if 'XGBoost' in nombre else y
        cv_res = cross_validate(
            pipe, X, y_target,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        resultados.append({
            'Modelo': nombre,
            'Accuracy Media': float(np.mean(cv_res['test_accuracy'])),
            'Accuracy Std': float(np.std(cv_res['test_accuracy'])),
            'Balanced Accuracy': float(np.mean(cv_res['test_balanced_accuracy'])),
            'Macro F1-Score': float(np.mean(cv_res['test_f1_macro'])),
            'Weighted F1-Score': float(np.mean(cv_res['test_f1_weighted']))
        })
        
    df_resultados = pd.DataFrame(resultados).sort_values(by='Macro F1-Score', ascending=False).reset_index(drop=True)
    return df_resultados

def entrenar_y_exportar_mejor_modelo(csv_path: str, models_dir: str):
    """
    Entrena el modelo ganador sobre un split de entrenamiento (80%) y evalúa
    en el test set independiente (20%). Luego guarda el modelo final entrenado con
    los metadatos de validación.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Carga cruda del dataset
    df_raw = pd.read_csv(csv_path)
    y_raw = df_raw['Severity']
    X_raw = df_raw.drop(columns=['Patient_ID', 'Diagnosis', 'Treatment_Plan', 'Severity'])
    
    # 2. Split estratificado Train/Test (80% / 20%)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    
    extractor = ClinicalFeatureExtractor()
    X_train = extractor.fit_transform(X_train_raw)
    X_test = extractor.transform(X_test_raw)
    
    # 3. Benchmark inicial
    print("Iniciando validación cruzada (5-Fold Stratified) de los modelos candidatos...")
    tabla_benchmark = benchmark_modelos(X_train, y_train)
    print("\n--- TABLA DE BENCHMARK (CV 5-FOLDS EN TRAIN) ---")
    print(tabla_benchmark.to_string(index=False))
    
    # 4. Configuración y Entrenamiento del Modelo Ganador
    # Random Forest es excelente para interpretabilidad clínica y captura de no-linealidades
    print("\nOptimizando y entrenando Random Forest Classifier final...")
    best_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=9,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    best_clf.fit(X_train, y_train)
    
    # 5. Evaluación en Conjunto de Prueba Independiente (Hold-Out Test)
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)
    
    clases_orden = list(best_clf.classes_)
    idx_severe = clases_orden.index('Severe') if 'Severe' in clases_orden else 0
    
    rep_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=clases_orden).tolist()
    
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    severe_recall = float(recall_score(y_test, y_pred, labels=['Severe'], average=None)[0])
    severe_f1 = float(f1_score(y_test, y_pred, labels=['Severe'], average=None)[0])
    
    print("\n--- RESULTADOS EN CONJUNTO DE PRUEBA (TEST SET INDEPENDIENTE, N=400) ---")
    print(f"Accuracy Global:      {acc*100:.2f}%")
    print(f"Balanced Accuracy:    {bal_acc*100:.2f}%")
    print(f"Macro F1-Score:       {macro_f1*100:.2f}%")
    print(f"Sensibilidad (Severe): {severe_recall*100:.2f}%")
    print(f"F1-Score (Severe):     {severe_f1*100:.2f}%")
    print("\nMatriz de Confusión (Filas=Real, Columnas=Predicción):")
    print(pd.DataFrame(conf_mat, index=[f"Real_{c}" for c in clases_orden], columns=[f"Pred_{c}" for c in clases_orden]))
    
    # 6. Importancia de Características (Feature Importance)
    feature_names = X_train.columns.tolist()
    importances = best_clf.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\n--- TOP 10 VARIABLES MÁS IMPORTANTES ---")
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:<26}: {imp*100:.2f}%")
        
    # 7. Empaquetar Pipeline Completo (FeatureExtractor + Classifier)
    # Permite inferencia directa recibiendo un DataFrame crudo con síntomas y signos vitales
    full_pipeline = Pipeline([
        ('feature_extractor', ClinicalFeatureExtractor()),
        ('classifier', best_clf)
    ])
    
    # Guardar modelo serializado
    model_file = os.path.join(models_dir, 'triaje_model.joblib')
    joblib.dump(full_pipeline, model_file)
    print(f"\nPipeline completo guardado en: {model_file}")
    
    # Guardar metadatos completos para documentación y reproducibilidad
    metadata = {
        'model_name': 'Random Forest Classifier (Triage Clinical Pipeline)',
        'target_classes': clases_orden,
        'feature_names': feature_names,
        'test_metrics': {
            'accuracy': round(acc, 4),
            'balanced_accuracy': round(bal_acc, 4),
            'macro_f1': round(macro_f1, 4),
            'severe_recall': round(severe_recall, 4),
            'severe_f1': round(severe_f1, 4),
            'classification_report': rep_dict,
            'confusion_matrix': {
                'labels': clases_orden,
                'matrix': conf_mat
            }
        },
        'benchmark_cv_5fold': tabla_benchmark.to_dict(orient='records'),
        'feature_importance_top10': [{'feature': f, 'importance': round(float(imp), 4)} for f, imp in feat_imp[:10]]
    }
    
    metadata_file = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"Metadatos del modelo guardados en: {metadata_file}")
    return metadata

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, 'data', 'disease_diagnosis.csv')
    models_path = os.path.join(base_dir, 'models')
    entrenar_y_exportar_mejor_modelo(csv_file, models_path)
