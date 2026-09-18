"""
Módulo cliente para el LLM Institucional de la Universidad de Cuenca (UCuenca).
Compatible con la API de Ollama y el Gateway Institucional.
"""

import os
import json
import ssl
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Generator

DEFAULT_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "https://mcp.ucuenca.edu.ec/llm/api")
DEFAULT_MODEL = "ministral-3:8b"

# Contexto SSL para certificados institucionales
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE


class UCuencaLLMClient:
    """Cliente para interactuar con los modelos LLM de UCuenca."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = DEFAULT_GATEWAY_URL):
        self.api_key = (api_key or os.getenv("LLM_API_KEY", "")).strip()
        self.base_url = base_url.rstrip("/")

    def get_models(self) -> List[str]:
        """Obtiene la lista de identificadores de modelos disponibles."""
        url = f"{self.base_url}/tags"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10, context=_ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        system: Optional[str] = None
    ) -> str:
        """Envía un prompt al endpoint /generate y retorna la respuesta de texto."""
        if not self.api_key:
            raise ValueError("Se requiere configurar la API Key (LLM_API_KEY).")

        url = f"{self.base_url}/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=90, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL
    ) -> str:
        """Envía una conversación de mensajes al endpoint /chat."""
        if not self.api_key:
            raise ValueError("Se requiere configurar la API Key (LLM_API_KEY).")

        url = f"{self.base_url}/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=90, context=_ssl_context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            message = data.get("message", {})
            return message.get("content", "")
