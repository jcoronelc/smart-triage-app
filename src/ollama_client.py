"""
Cliente de Conexión a Ollama y Motor de Generación Graph RAG
Proyecto: Triaje Predictivo en Urgencias
Permite conectar tanto a Ollama local (localhost:11434) como a un servidor remoto
(computador expuesto vía túnel Ngrok / Cloudflare / IP pública).
"""

import os
import json
import requests
from typing import Dict, List, Any, Generator, Tuple

# URL primaria configurada por el usuario (túnel para despliegue remoto)
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_SERVER_URL", "https://icy-dogs-sin.loca.lt")
FALLBACK_LOCAL_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"

TUNNEL_HEADERS = {
    "Bypass-Tunnel-Reminder": "true",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

def verificar_conexion_ollama(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.5) -> Tuple[bool, List[str], str]:
    """
    Comprueba si el servidor de Ollama responde en la URL indicada y recupera
    la lista de modelos instalados en la máquina. Si la URL de túnel no responde
    o está protegida, intenta automáticamente conectar al host local.
    """
    urls_to_try = [base_url]
    if base_url != FALLBACK_LOCAL_URL:
        urls_to_try.append(FALLBACK_LOCAL_URL)
        
    for target_url in urls_to_try:
        url_limpia = target_url.rstrip('/')
        endpoint = f"{url_limpia}/api/tags"
        try:
            r = requests.get(endpoint, headers=TUNNEL_HEADERS, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                modelos = [m['name'] for m in data.get('models', [])]
                return True, modelos, f"Conexión exitosa con el servidor Ollama ({target_url})."
        except Exception:
            continue
            
    return False, [], f"No se pudo conectar al servidor Ollama en {base_url}."

def construir_prompt_sistema_graph_rag(graph_context: Dict[str, Any], patient_summary: Dict[str, Any]) -> str:
    """
    Crea el System Prompt que ancla al LLM a la ontología del grafo de conocimiento,
    previniendo alucinaciones médicas.
    """
    precauciones_str = ", ".join(graph_context.get('precauciones_validadas', [])) or "Seguimiento médico general"
    farmacos_str = ", ".join(graph_context.get('farmacos_ontologia', [])) or "Medicación sintomática bajo prescripción"
    dietas_str = ", ".join(graph_context.get('dietas_ontologia', [])) or "Hidratación adecuada y reposo"
    relaciones_str = "\n".join([f"  • {rel}" for rel in graph_context.get('relaciones_grafo_evidencia', [])[:6]])
    
    system_prompt = f"""Eres el Asistente Clínico Inteligente del Servicio de Urgencias Hospitalarias, basado en una arquitectura Graph RAG (Grafo de Conocimiento Médico + Razonamiento LLM).

Tu misión es brindar orientación clínica clara, profesional y empática al personal médico o al paciente, explicando la severidad del triaje, las precauciones validadas y los signos de alarma.

=== DATOS DEL PACIENTE EVALUADO ===
- Edad: {patient_summary.get('Edad', 'No especificada')} años | Sexo: {patient_summary.get('Sexo', 'No especificado')}
- Constantes Vitales: Frecuencia Cardíaca {patient_summary.get('FC', 'N/D')} lpm, Temperatura {patient_summary.get('Temp', 'N/D')} °C, Presión Arterial {patient_summary.get('PA', 'N/D')} mmHg, Saturación O2 {patient_summary.get('SatO2', 'N/D')} %
- Síntomas Manifestados: {patient_summary.get('Sintomas', 'No especificados')}
- Clasificación de Triaje (Modelo ML): {patient_summary.get('Triaje', 'No evaluado')} (Certeza: {patient_summary.get('Confianza', 'N/D')})

=== CONOCIMIENTO ESTRUCTURADO RECUPERADO DEL GRAFO (GRAPH RAG - FUENTE DE VERDAD) ===
- Nivel de Acción de Triaje: {graph_context.get('accion_triaje', '')}
- Precauciones Clínicas Validadas en Ontología: {precauciones_str}
- Fármacos / Terapias Asociadas en el Grafo: {farmacos_str}
- Recomendaciones Nutricionales del Grafo: {dietas_str}
- Hechos y Relaciones de Evidencia del Grafo:
{relaciones_str}

REGLAS BIOÉTICAS Y CLÍNICAS ESTRICTAS:
1. Respeta fielmente la severidad de triaje calculada por el modelo ({patient_summary.get('Triaje', 'Leve')}).
2. Si el triaje es SEVERO (Severe), enfatiza de manera prioritaria la necesidad de ATENCIÓN MÉDICA INMEDIATA y describe los signos de alarma que requieren asistencia presencial de urgencia.
3. Tus recomendaciones de precauciones y cuidados deben estar estrictamente ancladas a las recuperadas del grafo. NUNCA inventes dosis farmacológicas ni prescribas tratamientos no validados en el contexto del grafo.
4. Responde en español formal, clínico, conciso y de fácil comprensión.
"""
    return system_prompt

def stream_chat_ollama(
    mensajes: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    system_prompt: str = ""
) -> Generator[str, None, None]:
    """
    Genera la respuesta del LLM en streaming (token por token) usando la API de Ollama (/api/chat).
    """
    url_limpia = base_url.rstrip('/')
    endpoint = f"{url_limpia}/api/chat"
    
    payload_messages = []
    if system_prompt:
        payload_messages.append({'role': 'system', 'content': system_prompt})
    
    for m in mensajes:
        payload_messages.append({'role': m['role'], 'content': m['content']})
        
    payload = {
        'model': model,
        'messages': payload_messages,
        'stream': True,
        'options': {
            'temperature': 0.3,  # Baja temperatura para respuestas deterministas y seguras
            'top_p': 0.9
        }
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=TUNNEL_HEADERS, stream=True, timeout=90)
        response.raise_for_status()
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    chunk = json.loads(line)
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.RequestException as e:
        # Si falló la URL remota y estamos en local, intentar con FALLBACK_LOCAL_URL
        if base_url != FALLBACK_LOCAL_URL:
            try:
                local_endpoint = f"{FALLBACK_LOCAL_URL}/api/chat"
                local_resp = requests.post(local_endpoint, json=payload, stream=True, timeout=90)
                local_resp.raise_for_status()
                for line in local_resp.iter_lines(decode_unicode=True):
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
                return
            except Exception:
                pass
        yield f"\n\n[Error de comunicación con el servidor Ollama ({base_url})]: {str(e)}"

def generar_respuesta_fallback_grafo(graph_context: Dict[str, Any], patient_summary: Dict[str, Any], query: str) -> str:
    """
    Generador determinista de respaldo cuando Ollama no está conectado o el servidor está apagado.
    Garantiza que la aplicación nunca falle y responda siempre con los hechos del Grafo de Conocimiento.
    """
    sev = patient_summary.get('Triaje', 'Mild')
    prec = ", ".join(graph_context.get('precauciones_validadas', [])) or "Reposo y seguimiento clínico"
    farm = ", ".join(graph_context.get('farmacos_ontologia', [])) or "Medicación sintomática según prescripción"
    diet = ", ".join(graph_context.get('dietas_ontologia', [])) or "Hidratación oral y descanso"
    
    if sev == 'Severe':
        alerta = "ALERTA MÁXIMA DE TRIAJE: El paciente presenta criterios de gravedad severa con riesgo de descompensación respiratoria o hemodinámica. Requiere valoración médica inmediata en urgencias."
    elif sev == 'Moderate':
        alerta = "PRIORIDAD MODERADA: Paciente con cuadro agudo febril/sistémico sin compromiso vital inmediato. Se sugiere atención médica prioritaria (< 60 minutos)."
    else:
        alerta = "PRIORIDAD LEVE: Paciente con constantes vitales dentro de rangos seguros. Cuadro clínico compatible con manejo ambulatorio en sala general."
        
    res = f"""**[Respuesta basada en Evidencia del Grafo de Conocimiento (Modo Determinista)]**

**Orientación de Triaje:**
{alerta}

**Precauciones validadas en Ontología Médica:**
• {prec}

**Fármacos / Manejo sintomático referenciado en el Grafo:**
• {farm}

**Pautas de hidratación y dieta:**
• {diet}

*(Nota: Para interacción conversacional en lenguaje natural con el LLM, asegúrese de que su servidor Ollama local esté activo en la URL configurada).*"""
    return res
