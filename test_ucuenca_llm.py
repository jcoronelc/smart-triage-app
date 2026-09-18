#!/usr/bin/env python3
"""
Script para conectar al LLM institucional de la Universidad de Cuenca (UCuenca).
Compatible con la API estilo Ollama del gateway institucional:
  - Base URL: https://mcp.ucuenca.edu.ec/llm/api
  - Endpoints: /tags, /generate, /chat, /embed
  - Autenticación: Bearer <LLM_API_KEY>
"""

import os
import sys
import json
import ssl
import argparse
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

def _load_env_file(filepath: str = ".env"):
    """Carga variables desde archivo .env si existe."""
    if not os.path.exists(filepath):
        # probar en directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, ".env")
        if os.path.exists(candidate):
            filepath = candidate
        else:
            return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

_load_env_file()

DEFAULT_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "https://mcp.ucuenca.edu.ec/llm/api")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "ministral-3:8b")

# Contexto SSL para certificados de red institucional
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE



def get_available_models(gateway_url: str = DEFAULT_GATEWAY_URL, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Consulta la lista de modelos publicados en el gateway de UCuenca."""
    url = f"{gateway_url.rstrip('/')}/tags"
    headers = {"Content-Type": "application/json"}
    if api_key:
        try:
            safe_key = api_key.strip().encode("ascii").decode("ascii")
            headers["Authorization"] = f"Bearer {safe_key}"
        except UnicodeEncodeError:
            pass

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("models", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[ERROR] Error al consultar modelos ({e.code}): {body}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[ERROR] No se pudo conectar al endpoint de tags: {e}", file=sys.stderr)
        return []


def generate_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    stream: bool = False,
    system: Optional[str] = None
) -> Dict[str, Any]:
    """Envía una petición al endpoint /generate del LLM UCuenca."""
    if not api_key:
        raise ValueError("Se requiere una API key (LLM_API_KEY). Solicítala en el portal institucional.")

    url = f"{gateway_url.rstrip('/')}/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    if system:
        payload["system"] = system

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}"
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context) as response:
            resp_body = response.read().decode("utf-8")
            return json.loads(resp_body)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Error de conexión con el LLM: {e}")


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    stream: bool = False
) -> Dict[str, Any]:
    """Envía una petición al endpoint /chat del LLM UCuenca."""
    if not api_key:
        raise ValueError("Se requiere una API key (LLM_API_KEY).")

    url = f"{gateway_url.rstrip('/')}/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}"
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            resp_body = response.read().decode("utf-8")
            return json.loads(resp_body)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Error de conexión con el LLM: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cliente de prueba para LLM UCuenca")
    parser.add_argument("--key", default=os.getenv("LLM_API_KEY"), help="API Key institucional (o variable LLM_API_KEY)")
    parser.add_argument("--url", default=DEFAULT_GATEWAY_URL, help=f"URL del Gateway (default: {DEFAULT_GATEWAY_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo a consultar (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", default="Hola, preséntate brevemente y dime en qué puedes ayudarme.", help="Prompt a enviar")
    parser.add_argument("--list-models", action="store_true", help="Solo listar modelos disponibles y salir")
    args = parser.parse_args()

    print("=" * 60)
    print("🤖 Conexión al Gateway LLM UCuenca")
    print(f"📡 URL Gateway: {args.url}")
    print("=" * 60)

    # 1. Listar modelos disponibles
    print("\n🔍 Consultando modelos publicados...")
    models = get_available_models(args.url, args.key)
    if models:
        print(f"✅ Se encontraron {len(models)} modelos disponibles:")
        for m in models:
            ctx = m.get("context_length", "N/A")
            details = m.get("details", {})
            param_size = details.get("parameter_size", "N/A")
            print(f"  • {m.get('name')} (Parámetros: {param_size}, Contexto: {ctx})")
    else:
        print("⚠️ No se pudieron listar los modelos.")

    if args.list_models:
        return

    # 2. Verificar API key para enviar petición
    api_key = args.key
    if not api_key and sys.stdin.isatty():
        try:
            val = input("\n🔑 Pega tu API Key institucional (o Enter para omitir): ").strip()
            if val:
                api_key = val
        except (KeyboardInterrupt, EOFError):
            pass

    if not api_key:
        print("\n⚠️ AVISO: No se detectó ninguna API Key.")
        print("Puedes proporcionarla de las siguientes formas:")
        print("  1) Como argumento: python3 test_ucuenca_llm.py --key \"<TU_API_KEY>\"")
        print("  2) Como variable de entorno: export LLM_API_KEY=\"<TU_API_KEY>\"")
        print("\nSolicita tu API Key desde el portal institucional: https://mcp.ucuenca.edu.ec/llm/access/")
        return

    # Validar si tiene puntos suspensivos (copiado truncado)
    if "…" in api_key or "..." in api_key:
        print("\n❌ Error: La API Key proporcionada parece estar incompleta o truncada (contiene '...').")
        print("Por favor copia la clave completa desde el portal institucional.")
        return

    # 3. Enviar prompt de prueba
    print(f"\n🚀 Enviando petición al modelo: '{args.model}'...")
    print(f"💬 Prompt: \"{args.prompt}\"")
    print("-" * 60)

    try:
        resultado = generate_completion(
            prompt=args.prompt,
            model=args.model,
            api_key=api_key,
            gateway_url=args.url
        )
        respuesta = resultado.get("response", "")
        print("\n📥 Respuesta del LLM UCuenca:")
        print(respuesta)
        print("-" * 60)
        duracion_ns = resultado.get("total_duration")
        if duracion_ns:
            duracion_s = duracion_ns / 1e9
            print(f"⏱️ Tiempo total: {duracion_s:.2f}s")
        eval_count = resultado.get("eval_count")
        if eval_count:
            print(f"📊 Tokens generados: {eval_count}")
    except RuntimeError as err:
        print(f"\n❌ Error al comunicarse con el modelo: {err}")


if __name__ == "__main__":
    main()
