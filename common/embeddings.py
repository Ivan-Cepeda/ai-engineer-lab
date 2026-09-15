"""
Generacion de embeddings, con cache en disco.

QUE ES UN EMBEDDING, EN UNA FRASE
  Es una lista larga de numeros que representa el SIGNIFICADO de un texto.
  Textos que quieren decir cosas parecidas dan listas de numeros parecidas.

POR QUE HAY UN CACHE
  Tres razones, y las tres importan:

  1. Plata. Cada texto que embebes se paga. Si corres el mismo ejercicio diez
     veces, sin cache pagas diez veces por el mismo texto.
  2. Tiempo. Leer del disco tarda milisegundos; pedirselo al servidor, segundos.
  3. Es una tecnica real. En un sistema de produccion los documentos se embeben
     UNA vez y se guardan. Solo se re-embeben los que cambian.

  El cache se guarda en la carpeta cache/, que esta en .gitignore.
"""

import hashlib
import json
import os

from openai import OpenAI

from common.config import PRECIOS_EMBEDDING, cargar_configuracion


CARPETA_DEL_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_CACHE = os.path.join(CARPETA_DEL_PROYECTO, "cache")

# Cuantos textos mandamos por pedido. Mandar de a uno es lentisimo; mandar
# miles de golpe hace que el pedido sea rechazado por tamano. 64 es un punto
# medio comodo.
TAMANO_DEL_LOTE = 64


def crear_cliente(proveedor=None, modelo_embedding=None):
    """
    Prepara la conexion y devuelve un diccionario con lo necesario.

    Es el mismo patron de M1: un diccionario simple con las tres o cuatro cosas
    que hacen falta, en vez de un objeto complicado.
    """
    datos = cargar_configuracion(proveedor, None, modelo_embedding)

    conexion = OpenAI(api_key=datos["clave"], base_url=datos["base_url"])

    return {
        "api": conexion,
        "modelo_chat": datos["modelo_chat"],
        "modelo_embedding": datos["modelo_embedding"],
        "proveedor": datos["proveedor"],
        "clave": datos["clave"],
        # Contadores para poder mostrar cuanto se uso el cache.
        "textos_embebidos": 0,
        "textos_desde_cache": 0,
    }


def crear_cliente_desde_argumentos(args):
    """Version que toma los datos de lo que se escribio en la terminal."""
    return crear_cliente(args.provider, getattr(args, "embedding_model", None))


def _nombre_de_archivo_de_cache(modelo, texto):
    """
    Convierte (modelo + texto) en un nombre de archivo unico y corto.

    Usamos una funcion hash: dado un texto, siempre devuelve la misma cadena
    de letras y numeros. Asi el mismo texto siempre busca el mismo archivo.

    El modelo entra en el hash a proposito: los embeddings de dos modelos
    distintos NO son comparables entre si, y seria un error grave mezclarlos.
    """
    contenido = modelo + "||" + texto
    huella = hashlib.sha256(contenido.encode("utf-8")).hexdigest()

    return os.path.join(CARPETA_CACHE, huella + ".json")


def _leer_del_cache(modelo, texto):
    """Devuelve el vector guardado, o None si nunca se guardo."""
    ruta = _nombre_de_archivo_de_cache(modelo, texto)

    if not os.path.exists(ruta):
        return None

    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except (json.JSONDecodeError, OSError):
        # Si el archivo quedo corrupto, lo ignoramos y lo volvemos a pedir.
        return None


def _guardar_en_cache(modelo, texto, vector):
    """Guarda el vector en disco para la proxima vez."""
    os.makedirs(CARPETA_CACHE, exist_ok=True)

    ruta = _nombre_de_archivo_de_cache(modelo, texto)

    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(vector, archivo)


def embeber(cliente, textos, usar_cache=True, mostrar=False):
    """
    Convierte una lista de textos en una lista de vectores.

    Devuelve una lista donde la posicion i es el vector del texto i.

    El flujo es:
      1. Para cada texto, mirar si ya esta en el cache.
      2. Juntar los que faltan y pedirlos al proveedor en lotes.
      3. Guardar los nuevos en el cache.
      4. Devolver todo en el orden original.

    El paso 4 es mas delicado de lo que parece: los textos que vienen del cache
    y los que vienen del servidor se mezclan, y hay que devolverlos en el mismo
    orden en que nos los pidieron.
    """
    modelo = cliente["modelo_embedding"]

    # Reservamos un lugar por cada texto. Los vamos llenando a medida que
    # conseguimos los vectores, desde el cache o desde el servidor.
    resultados = [None] * len(textos)

    # Los que hay que pedirle al proveedor, con su posicion original.
    faltantes = []
    posiciones_faltantes = []

    for posicion in range(len(textos)):
        texto = textos[posicion]

        if usar_cache:
            vector_guardado = _leer_del_cache(modelo, texto)
        else:
            vector_guardado = None

        if vector_guardado is not None:
            resultados[posicion] = vector_guardado
            cliente["textos_desde_cache"] = cliente["textos_desde_cache"] + 1
        else:
            faltantes.append(texto)
            posiciones_faltantes.append(posicion)

    if mostrar:
        print("  [embeddings] " + str(len(textos)) + " textos: " +
              str(len(textos) - len(faltantes)) + " del cache, " +
              str(len(faltantes)) + " para pedir")

    # Pedimos los faltantes en lotes.
    inicio = 0

    while inicio < len(faltantes):
        lote = faltantes[inicio:inicio + TAMANO_DEL_LOTE]

        respuesta = cliente["api"].embeddings.create(model=modelo, input=lote)

        for numero_en_el_lote in range(len(lote)):
            vector = respuesta.data[numero_en_el_lote].embedding
            posicion_original = posiciones_faltantes[inicio + numero_en_el_lote]

            resultados[posicion_original] = vector

            if usar_cache:
                _guardar_en_cache(modelo, lote[numero_en_el_lote], vector)

        cliente["textos_embebidos"] = cliente["textos_embebidos"] + len(lote)
        inicio = inicio + TAMANO_DEL_LOTE

    return resultados


def embeber_uno(cliente, texto, usar_cache=True):
    """Atajo para cuando solo queres embeber un texto."""
    return embeber(cliente, [texto], usar_cache)[0]


def estimar_costo(modelo, cantidad_de_textos, palabras_promedio=50):
    """
    Estima cuanto costo embeber. Devuelve None si no conocemos el precio.

    Es una estimacion, no el numero exacto: la API de embeddings no siempre
    devuelve el conteo de tokens. Usamos la regla practica de 1 token cada
    0.75 palabras aproximadamente.
    """
    if modelo not in PRECIOS_EMBEDDING:
        return None

    tokens_estimados = cantidad_de_textos * palabras_promedio / 0.75

    return tokens_estimados / 1000000 * PRECIOS_EMBEDDING[modelo]


def limpiar_cache():
    """Borra todo el cache. Util para medir cuanto tarda sin cache."""
    if not os.path.exists(CARPETA_CACHE):
        return 0

    borrados = 0

    for nombre in os.listdir(CARPETA_CACHE):
        if nombre.endswith(".json"):
            os.remove(os.path.join(CARPETA_CACHE, nombre))
            borrados = borrados + 1

    return borrados
