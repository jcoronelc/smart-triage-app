"""
Cliente de Conexión al LLM Institucional UCuenca y Motor de Generación Graph RAG
Proyecto: Triaje Predictivo en Urgencias
Conecta al Gateway institucional de la Universidad de Cuenca compatible con API Ollama:
  - Base URL: https://mcp.ucuenca.edu.ec/llm/api
  - Endpoints: /tags, /chat, /generate
  - Autenticación: Bearer <LLM_API_KEY>
"""

import os
import json
import requests
import urllib3
from typing import Dict, List, Any, Generator, Tuple

# Suprimir advertencias de certificados autofirmados/institucionales
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def _load_env():
    """Carga configuración desde .env si existe."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass

_load_env()

# Configuración institucional por defecto
DEFAULT_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "https://mcp.ucuenca.edu.ec/llm/api")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "llm_hFaExr6x9e0PHtTkUNSicw0-UrnguGdFs8_fSVjnaP8")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "ministral-3:8b")

# Aliases de compatibilidad con código existente
DEFAULT_OLLAMA_URL = DEFAULT_GATEWAY_URL
FALLBACK_LOCAL_URL = "http://localhost:11434"

GATEWAY_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Streamlit-Triage-App/1.0"
}


def verificar_conexion_ollama(
    base_url: str = DEFAULT_GATEWAY_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = 3.0
) -> Tuple[bool, List[str], str]:
    """
    Comprueba si el Gateway institucional UCuenca o el servidor Ollama responde
    y recupera la lista de modelos disponibles.
    Retorna: (is_connected, list_of_models, active_url)
    """
    candidate_urls = []
    if base_url:
        candidate_urls.append(base_url.rstrip('/'))
    if DEFAULT_GATEWAY_URL not in candidate_urls:
        candidate_urls.append(DEFAULT_GATEWAY_URL.rstrip('/'))
    if FALLBACK_LOCAL_URL not in candidate_urls:
        candidate_urls.append(FALLBACK_LOCAL_URL)

    headers = dict(GATEWAY_HEADERS)
    active_key = api_key or os.getenv("LLM_API_KEY", DEFAULT_API_KEY)
    if active_key:
        headers["Authorization"] = f"Bearer {active_key.strip()}"

    for target_url in candidate_urls:
        # Probar endpoints estándar de Ollama y Gateway UCuenca
        for endpoint in [f"{target_url}/tags", f"{target_url}/api/tags"]:
            try:
                r = requests.get(endpoint, headers=headers, timeout=timeout, verify=False)
                if r.status_code == 200:
                    data = r.json()
                    modelos = [m['name'] for m in data.get('models', []) if 'name' in m]
                    if modelos:
                        return True, modelos, target_url
            except Exception:
                continue

    return False, [], base_url


def construir_prompt_sistema_graph_rag(graph_context: Dict[str, Any], patient_summary: Dict[str, Any]) -> str:
    """
    Crea el System Prompt que ancla al LLM a la ontología del grafo de conocimiento,
    previniendo alucinaciones médicas.
    """
    precauciones_str = ", ".join(graph_context.get('precauciones_validadas', [])) or "Seguimiento médico general"
    farmacos_str = ", ".join(graph_context.get('farmacos_ontologia', [])) or "Medicación sintomática bajo prescripción"
    dietas_str = ", ".join(graph_context.get('dietas_ontologia', [])) or "Hidratación adecuada y reposo"
    relaciones_str = "\n".join([f"  • {rel}" for rel in graph_context.get('relaciones_grafo_evidencia', [])[:6]])

    system_prompt = f"""Eres el Asistente Clínico Inteligente del Servicio de Urgencias Hospitalarias, basado en una arquitectura Graph RAG (Grafo de Conocimiento Médico + Razonamiento LLM de la Universidad de Cuenca).

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
    base_url: str = DEFAULT_GATEWAY_URL,
    api_key: str = DEFAULT_API_KEY,
    system_prompt: str = ""
) -> Generator[str, None, None]:
    """
    Genera la respuesta del LLM en streaming (token por token) usando la API del Gateway UCuenca / Ollama (/chat o /api/chat).
    """
    payload_messages = []
    if system_prompt:
        payload_messages.append({'role': 'system', 'content': system_prompt})

    for m in mensajes:
        if m.get('content'):
            payload_messages.append({'role': m['role'], 'content': m['content']})

    payload = {
        'model': model or DEFAULT_MODEL,
        'messages': payload_messages,
        'stream': True,
        'options': {
            'temperature': 0.3,
            'top_p': 0.9
        }
    }

    headers = dict(GATEWAY_HEADERS)
    active_key = api_key or os.getenv("LLM_API_KEY", DEFAULT_API_KEY)
    if active_key:
        headers["Authorization"] = f"Bearer {active_key.strip()}"

    candidate_urls = []
    if base_url:
        candidate_urls.append(base_url.rstrip('/'))
    if DEFAULT_GATEWAY_URL.rstrip('/') not in candidate_urls:
        candidate_urls.append(DEFAULT_GATEWAY_URL.rstrip('/'))

    streamed_any = False
    last_err = None

    for target_url in candidate_urls:
        # Probar endpoints /chat y /api/chat
        for endpoint_name in ["/chat", "/api/chat"]:
            endpoint = f"{target_url}{endpoint_name}"
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(5.0, 90.0),
                    verify=False
                )
                if response.status_code != 200:
                    last_err = f"HTTP {response.status_code}: {response.text[:120]}"
                    continue

                for raw_line in response.iter_lines():
                    if raw_line:
                        if isinstance(raw_line, bytes):
                            line = raw_line.decode('utf-8', errors='ignore')
                        else:
                            line = raw_line
                        try:
                            chunk = json.loads(line)
                            # Soporta formato Ollama (/chat) y (/generate)
                            content = chunk.get('message', {}).get('content') or chunk.get('response', '')
                            if content:
                                streamed_any = True
                                yield content
                        except json.JSONDecodeError:
                            continue

                if streamed_any:
                    return
            except Exception as e:
                last_err = e
                continue

    if not streamed_any:
        yield f"\n\n*(Aviso: No se pudo recibir respuesta del modelo `{model}` en {base_url}. Error: {last_err or 'Servidor no disponible'}).*"


def generar_respuesta_fallback_grafo(graph_context: Dict[str, Any], patient_summary: Dict[str, Any], query: str) -> str:
    """
    Generador determinista de respaldo cuando el LLM no está conectado.
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

*(Nota: Grafo de conocimiento ontológico activo. Conexión al Gateway LLM UCuenca en espera).*"""
    return res


def stream_fallback_grafo(graph_context: Dict[str, Any], patient_summary: Dict[str, Any], query: str) -> Generator[str, None, None]:
    """Genera la respuesta del grafo en streaming token por token con un efecto fluido."""
    import time
    respuesta_completa = generar_respuesta_fallback_grafo(graph_context, patient_summary, query)
    tokens = respuesta_completa.split(" ")
    for tok in tokens:
        yield tok + " "
        time.sleep(0.012)
