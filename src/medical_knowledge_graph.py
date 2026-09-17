"""
Módulo de Grafo de Conocimiento Médico (Graph RAG)
Proyecto: Triaje Predictivo y Asistente Clínico Inteligente
Construye una ontología clínica estructurada que ancla las respuestas del LLM
a hechos médicos verificados (descripciones, precauciones, fármacos y dietas).
"""

import os
import ast
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'kaggle_dataset')

# Mapeo de enfermedades del modelo de triaje con las del grafo de conocimiento
MAPEO_ENFERMEDADES = {
    'Bronchitis': 'Bronchial Asthma',
    'Pneumonia': 'Pneumonia',
    'Flu': 'Common Cold',  # Influenza / Cuadro viral agudo
    'Cold': 'Common Cold',
    'Healthy': 'Healthy'
}

class MedicalKnowledgeGraph:
    """
    Grafo de conocimiento médico para Graph RAG.
    Almacena nodos de tipo:
    - Disease (Enfermedad)
    - Symptom (Síntoma)
    - Precaution (Precaución clínica)
    - Medication (Medicamento)
    - Diet (Dieta/Nutrición)
    - Severity (Nivel de Severidad de Triaje)
    """
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.graph = nx.DiGraph()
        self._construir_grafo()

    def _safe_parse_list(self, val: Any) -> List[str]:
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass
            return [x.strip() for x in val.replace('[', '').replace(']', '').replace("'", "").split(',') if x.strip()]
        return []

    def _construir_grafo(self):
        # 1. Cargar descripciones
        desc_path = os.path.join(self.data_dir, 'description.csv')
        if os.path.exists(desc_path):
            df_desc = pd.read_csv(desc_path)
            for _, row in df_desc.iterrows():
                disease = str(row['Disease']).strip()
                self.graph.add_node(disease, type='Disease', description=str(row['Description']).strip())

        # 2. Cargar precauciones
        prec_path = os.path.join(self.data_dir, 'precautions_df.csv')
        if os.path.exists(prec_path):
            df_prec = pd.read_csv(prec_path)
            for _, row in df_prec.iterrows():
                disease = str(row['Disease']).strip()
                if not self.graph.has_node(disease):
                    self.graph.add_node(disease, type='Disease', description="Condición respiratoria / clínica.")
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in row and pd.notna(row[col]):
                        prec_text = str(row[col]).strip().capitalize()
                        prec_node = f"Precaucion: {prec_text}"
                        self.graph.add_node(prec_node, type='Precaution', text=prec_text)
                        self.graph.add_edge(disease, prec_node, relation='REQUIRES_PRECAUTION')

        # 3. Cargar medicamentos
        med_path = os.path.join(self.data_dir, 'medications.csv')
        if os.path.exists(med_path):
            df_med = pd.read_csv(med_path)
            for _, row in df_med.iterrows():
                disease = str(row['Disease']).strip()
                meds = self._safe_parse_list(row['Medication'])
                for m in meds:
                    med_node = f"Farmaco: {m}"
                    self.graph.add_node(med_node, type='Medication', name=m)
                    self.graph.add_edge(disease, med_node, relation='RECOMMENDS_MEDICATION')

        # 4. Cargar dietas
        diet_path = os.path.join(self.data_dir, 'diets.csv')
        if os.path.exists(diet_path):
            df_diet = pd.read_csv(diet_path)
            for _, row in df_diet.iterrows():
                disease = str(row['Disease']).strip()
                diets = self._safe_parse_list(row['Diet'])
                for d in diets:
                    diet_node = f"Dieta: {d}"
                    self.graph.add_node(diet_node, type='Diet', name=d)
                    self.graph.add_edge(disease, diet_node, relation='SUGGESTS_DIET')

        # 5. Agregar relaciones ontológicas de triaje (Severity)
        niveles_severidad = {
            'Severe': 'Atención Médica Inmediata requerida en Box de Vitales / Reanimación',
            'Moderate': 'Atención Prioritaria en Sala de Observación (< 60 min)',
            'Mild': 'Atención Ambulatoria Estándar / Sala de Espera General'
        }
        for sev, desc in niveles_severidad.items():
            self.graph.add_node(f"Severidad_{sev}", type='Severity', level=sev, clinical_action=desc)

        # Enlazar enfermedades a su nivel de severidad típico
        asociaciones_triaje = {
            'Pneumonia': 'Severe',
            'Bronchial Asthma': 'Severe',
            'Bronchitis': 'Severe',
            'Common Cold': 'Mild'
        }
        for dis, sev in asociaciones_triaje.items():
            if self.graph.has_node(dis):
                self.graph.add_edge(dis, f"Severidad_{sev}", relation='HAS_TRIAGE_SEVERITY')

    def retrieve_clinical_context(self, severity: str, symptoms: Optional[List[str]] = None, suspected_disease: Optional[str] = None) -> Dict[str, Any]:
        """
        Recuperador Graph RAG: Extrae del grafo las entidades y relaciones estructuradas
        asociadas a la severidad predicha y los síntomas del paciente.
        """
        # Determinar enfermedades relevantes en el grafo
        if not suspected_disease:
            if severity == 'Severe':
                candidate_diseases = ['Pneumonia', 'Bronchial Asthma']
            elif severity == 'Moderate':
                candidate_diseases = ['Common Cold', 'Bronchial Asthma']
            else:
                candidate_diseases = ['Common Cold']
        else:
            mapped = MAPEO_ENFERMEDADES.get(suspected_disease, suspected_disease)
            candidate_diseases = [mapped] if mapped in self.graph else ['Common Cold']

        evidencias = []
        todas_precauciones = []
        todos_farmacos = []
        todas_dietas = []
        descripciones = []

        for dis in candidate_diseases:
            if not self.graph.has_node(dis):
                continue
            
            dis_data = self.graph.nodes[dis]
            desc = dis_data.get('description', '')
            descripciones.append(f"{dis}: {desc}")

            # Buscar precauciones
            for nbr in self.graph.successors(dis):
                edge_data = self.graph.get_edge_data(dis, nbr)
                rel = edge_data.get('relation', '')
                node_type = self.graph.nodes[nbr].get('type', '')

                if rel == 'REQUIRES_PRECAUTION' or node_type == 'Precaution':
                    txt = self.graph.nodes[nbr].get('text', str(nbr).replace('Precaucion: ', ''))
                    if txt not in todas_precauciones:
                        todas_precauciones.append(txt)
                        evidencias.append(f"({dis}) -[:REQUIRES_PRECAUTION]-> ({txt})")

                elif rel == 'RECOMMENDS_MEDICATION' or node_type == 'Medication':
                    med_name = self.graph.nodes[nbr].get('name', str(nbr).replace('Farmaco: ', ''))
                    if med_name not in todos_farmacos:
                        todos_farmacos.append(med_name)
                        evidencias.append(f"({dis}) -[:RECOMMENDS_MEDICATION]-> ({med_name})")

                elif rel == 'SUGGESTS_DIET' or node_type == 'Diet':
                    diet_name = self.graph.nodes[nbr].get('name', str(nbr).replace('Dieta: ', ''))
                    if diet_name not in todas_dietas:
                        todas_dietas.append(diet_name)
                        evidencias.append(f"({dis}) -[:SUGGESTS_DIET]-> ({diet_name})")

        # Nivel de acción clínica según triaje
        accion_triaje = {
            'Severe': 'PRIORIDAD MÁXIMA: Paciente en riesgo de deterioro hemodinámico/respiratorio. Requiere atención médica inmediata.',
            'Moderate': 'PRIORIDAD MEDIA: Paciente con compromiso sistémico o febril agudo. Ubicación en observación (<60 min).',
            'Mild': 'PRIORIDAD ESTÁNDAR: Constantes fisiológicas normales. Paciente apto para atención ambulatoria en sala de espera.'
        }.get(severity, 'Evaluación estándar.')

        return {
            'severidad': severity,
            'accion_triaje': accion_triaje,
            'enfermedades_consultadas': candidate_diseases,
            'descripciones': descripciones,
            'precauciones_validadas': todas_precauciones[:6],
            'farmacos_ontologia': todos_farmacos[:6],
            'dietas_ontologia': todas_dietas[:6],
            'relaciones_grafo_evidencia': evidencias[:12],
            'resumen_grafo': {
                'total_nodos': self.graph.number_of_nodes(),
                'total_aristas': self.graph.number_of_edges()
            }
        }

# Instancia global del grafo
_instancia_grafo: Optional[MedicalKnowledgeGraph] = None

def obtener_grafo_medico() -> MedicalKnowledgeGraph:
    global _instancia_grafo
    if _instancia_grafo is None:
        _instancia_grafo = MedicalKnowledgeGraph()
    return _instancia_grafo

if __name__ == '__main__':
    kg = obtener_grafo_medico()
    print(f"Grafo cargado: {kg.graph.number_of_nodes()} nodos, {kg.graph.number_of_edges()} aristas.")
    ctx = kg.retrieve_clinical_context('Severe', suspected_disease='Pneumonia')
    print("\n--- CONTEXTO RECUPERADO DEL GRAFO PARA PNEUMONIA / SEVERE ---")
    print("Acción de Triaje:", ctx['accion_triaje'])
    print("Precauciones:", ctx['precauciones_validadas'])
    print("Fármacos:", ctx['farmacos_ontologia'])
    print("Relaciones de Evidencia:", ctx['relaciones_grafo_evidencia'][:4])
