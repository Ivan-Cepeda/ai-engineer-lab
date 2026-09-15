"""
Configuracion y credenciales del modulo 2.

Es la misma idea que en M1, con un agregado: ahora no alcanza con saber que
modelo de CHAT usamos. Tambien hace falta el modelo de EMBEDDINGS, que es un
modelo distinto, con su propio nombre y su propio precio.

Regla de oro, igual que siempre: las claves nunca se escriben en el codigo.
Viven en un archivo .env que no se sube al repositorio.
"""

import argparse
import os

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Catalogo de proveedores.
#
# Cada proveedor tiene DOS modelos:
#   modelo_chat      -> el que escribe texto (lo usamos en L3, para RAG)
#   modelo_embedding -> el que convierte texto en numeros (L1, L2 y L4)
#
# El truco que permite usar los dos proveedores con el mismo codigo es el
# mismo de M1: la libreria "openai" habla un protocolo, y Google publica ese
# mismo protocolo en una direccion de compatibilidad. Y eso incluye el
# endpoint de embeddings, no solo el de chat.
# ---------------------------------------------------------------------------
PROVEEDORES = {
    "openai": {
        "variable_clave": "OPENAI_API_KEY",
        "modelo_chat": "gpt-4o-mini",
        "modelo_embedding": "text-embedding-3-small",
        "dimensiones": 1536,
        "base_url": None,   # None = la direccion de fabrica del SDK
        "donde_conseguir_clave": "https://platform.openai.com/api-keys",
    },
    "gemini": {
        "variable_clave": "GEMINI_API_KEY",
        "modelo_chat": "gemini-3.7-flash",
        "modelo_embedding": "gemini-embedding-001",
        "dimensiones": 3072,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "donde_conseguir_clave": "https://aistudio.google.com/apikey",
    },
}


# Precios de los modelos de EMBEDDINGS, en dolares por millon de tokens.
# Verificados en septiembre de 2026.
#
# Fijate lo baratos que son comparados con los modelos de chat: embeber texto
# cuesta centavos. Lo caro de un sistema RAG casi nunca es la parte de
# embeddings, es la llamada de generacion del final.
PRECIOS_EMBEDDING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "gemini-embedding-001": 0.15,
}


class ErrorDeConfiguracion(Exception):
    """Se lanza cuando falta una credencial o el proveedor no existe."""


def cargar_configuracion(proveedor=None, modelo_chat=None, modelo_embedding=None):
    """
    Resuelve con que proveedor y con que modelos vamos a trabajar.

    Prioridad: lo que se pasa por linea de comandos, despues el .env, y por
    ultimo los valores por defecto de este archivo.
    """
    load_dotenv()

    if proveedor is None:
        proveedor = os.getenv("LLM_PROVIDER")
    if proveedor is None:
        proveedor = "openai"

    proveedor = proveedor.strip().lower()

    if proveedor not in PROVEEDORES:
        raise ErrorDeConfiguracion(
            "Proveedor desconocido: '" + proveedor + "'. Opciones: openai, gemini"
        )

    datos = PROVEEDORES[proveedor]
    clave = os.getenv(datos["variable_clave"])

    if not clave:
        raise ErrorDeConfiguracion(
            "\nFalta la variable " + datos["variable_clave"] + " para usar " + proveedor + ".\n"
            "  1) Copia .env.example a .env\n"
            "  2) Pega tu clave en " + datos["variable_clave"] + "=\n"
            "  3) Conseguis una clave en: " + datos["donde_conseguir_clave"] + "\n"
        )

    # Los modelos se pueden sobreescribir desde el .env o desde la terminal.
    if modelo_chat is None:
        modelo_chat = os.getenv(proveedor.upper() + "_CHAT_MODEL")
    if modelo_chat is None:
        modelo_chat = datos["modelo_chat"]

    if modelo_embedding is None:
        modelo_embedding = os.getenv(proveedor.upper() + "_EMBEDDING_MODEL")
    if modelo_embedding is None:
        modelo_embedding = datos["modelo_embedding"]

    return {
        "proveedor": proveedor,
        "clave": clave,
        "modelo_chat": modelo_chat,
        "modelo_embedding": modelo_embedding,
        "base_url": datos["base_url"],
    }


def crear_parser(descripcion):
    """Argumentos comunes a todos los ejercicios del modulo."""
    parser = argparse.ArgumentParser(description=descripcion)

    parser.add_argument(
        "--provider",
        choices=sorted(PROVEEDORES),
        default=None,
        help="Proveedor a usar. Por defecto toma LLM_PROVIDER del .env",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Modelo de embeddings especifico",
    )
    parser.add_argument(
        "--sin-cache",
        action="store_true",
        help="Ignorar el cache y volver a pedirle los embeddings al proveedor",
    )
    return parser
