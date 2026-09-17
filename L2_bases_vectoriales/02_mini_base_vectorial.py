"""
L2 - Ejercicio 02: una base de datos vectorial completa, en 150 lineas.

QUE VAMOS A CONSTRUIR
  Una base vectorial de verdad, con todo lo que tiene una real:

    * agregar documentos
    * guardar y cargar desde disco (persistencia)
    * buscar por similitud
    * filtrar por metadata antes de buscar
    * actualizar y borrar
    * verificar que las dimensiones coincidan

  Despues de esto, cuando leas la documentacion de Pinecone o Qdrant, vas a
  reconocer cada operacion.

POR QUE ESCRIBIRLA EN VEZ DE USAR UNA
  Porque las bases vectoriales parecen magia hasta que ves que son un archivo
  con vectores y una multiplicacion de matrices. Y porque para menos de
  50.000 documentos, esto ALCANZA. No siempre hace falta un servidor.

COMO CORRERLO
    python 02_mini_base_vectorial.py
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.chunking import chunking_respetando_limites, crear_chunks_con_metadata
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.ui import mostrar_configuracion, subtitulo, titulo
from common.vectores import normalizar_matriz


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO_DOC = os.path.join(CARPETA, "datos", "base_conocimiento.md")
CARPETA_INDICES = os.path.join(CARPETA, "indices")


# ---------------------------------------------------------------------------
# LA BASE VECTORIAL
#
# La representamos con un diccionario que tiene tres cosas:
#   "vectores"  -> la matriz de NumPy, una fila por documento
#   "registros" -> la lista de metadata, en el MISMO orden que las filas
#   "modelo"    -> con que modelo se generaron, para no mezclar
#
# Que el orden coincida entre la matriz y la lista es lo que hace que todo
# funcione: la fila 5 de la matriz corresponde al registro 5 de la lista.
# ---------------------------------------------------------------------------

def crear_base(modelo):
    """Crea una base vacia."""
    return {"vectores": None, "registros": [], "modelo": modelo}


def agregar(base, vectores_nuevos, registros_nuevos):
    """
    Agrega documentos a la base.

    LA VERIFICACION DE DIMENSIONES ES LO MAS IMPORTANTE DE ESTA FUNCION.
    Si alguien agrega vectores de un modelo distinto, la base queda corrupta:
    las busquedas van a devolver resultados sin sentido y no vas a entender por
    que. Mejor fallar ruidosamente ahora que en silencio dentro de dos meses.
    """
    if len(vectores_nuevos) != len(registros_nuevos):
        raise ValueError(
            "Llegaron " + str(len(vectores_nuevos)) + " vectores y " +
            str(len(registros_nuevos)) + " registros. Tiene que haber uno por uno."
        )

    matriz_nueva = np.array(vectores_nuevos, dtype=np.float32)

    if base["vectores"] is None:
        base["vectores"] = matriz_nueva
    else:
        dimensiones_actuales = base["vectores"].shape[1]
        dimensiones_nuevas = matriz_nueva.shape[1]

        if dimensiones_actuales != dimensiones_nuevas:
            raise ValueError(
                "La base guarda vectores de " + str(dimensiones_actuales) +
                " dimensiones y estas agregando de " + str(dimensiones_nuevas) + ".\n"
                "Causa casi segura: cambiaste de modelo de embeddings. Los "
                "vectores de modelos distintos NO son comparables: hay que "
                "volver a embeber todo con el modelo nuevo."
            )

        # vstack apila la matriz nueva debajo de la que ya teniamos.
        base["vectores"] = np.vstack([base["vectores"], matriz_nueva])

    for registro in registros_nuevos:
        base["registros"].append(registro)

    return base


def buscar(base, vector_consulta, k=3, filtro=None):
    """
    Busca los k documentos mas parecidos, con filtro opcional por metadata.

    EL FILTRO ES LA FUNCION QUE MAS SE SUBESTIMA. En un sistema real casi nunca
    queres buscar en TODO: queres buscar en los documentos de este cliente, de
    este ano, de esta categoria. Filtrar primero y buscar despues es mas rapido
    y mas correcto.

    "filtro" es una funcion que recibe un registro y devuelve True o False.
    """
    if base["vectores"] is None or len(base["registros"]) == 0:
        return []

    consulta = np.array(vector_consulta, dtype=np.float32)

    if consulta.shape[0] != base["vectores"].shape[1]:
        raise ValueError(
            "La consulta tiene " + str(consulta.shape[0]) + " dimensiones y la "
            "base " + str(base["vectores"].shape[1]) + ". Embebiste la consulta "
            "con otro modelo?"
        )

    # Decidimos que filas participan de la busqueda.
    if filtro is None:
        posiciones_candidatas = list(range(len(base["registros"])))
    else:
        posiciones_candidatas = []
        for posicion in range(len(base["registros"])):
            if filtro(base["registros"][posicion]):
                posiciones_candidatas.append(posicion)

    if len(posiciones_candidatas) == 0:
        return []

    # Nos quedamos solo con las filas candidatas.
    submatriz = base["vectores"][posiciones_candidatas]

    # Normalizamos las dos partes para que el producto sea la similitud coseno.
    submatriz = normalizar_matriz(submatriz)
    longitud = np.linalg.norm(consulta)
    if longitud > 0:
        consulta = consulta / longitud

    puntajes = submatriz @ consulta

    # argsort ordena de menor a mayor; [::-1] lo da vuelta.
    orden = np.argsort(puntajes)[::-1]

    resultados = []

    for indice_en_la_submatriz in orden[:k]:
        posicion_real = posiciones_candidatas[indice_en_la_submatriz]

        resultados.append({
            "registro": base["registros"][posicion_real],
            "puntaje": float(puntajes[indice_en_la_submatriz]),
            "posicion": posicion_real,
        })

    return resultados


def borrar(base, chunk_id):
    """
    Borra un documento por su id.

    Reconstruye la matriz sin esa fila. En una base real esto seria un borrado
    logico (marcarlo como borrado y limpiar despues), porque reconstruir la
    matriz entera cada vez que borras uno es caro. Aca lo hacemos simple.
    """
    posiciones_a_conservar = []

    for posicion in range(len(base["registros"])):
        if base["registros"][posicion]["chunk_id"] != chunk_id:
            posiciones_a_conservar.append(posicion)

    if len(posiciones_a_conservar) == len(base["registros"]):
        return False   # no habia nada con ese id

    base["vectores"] = base["vectores"][posiciones_a_conservar]

    registros_nuevos = []
    for posicion in posiciones_a_conservar:
        registros_nuevos.append(base["registros"][posicion])

    base["registros"] = registros_nuevos

    return True


def guardar(base, nombre):
    """
    Guarda la base en disco: los vectores en .npy y la metadata en .json.

    Separamos los dos formatos a proposito. El .npy guarda numeros de forma
    compacta y se carga rapidisimo; el .json se puede abrir y leer con
    cualquier editor cuando necesites revisar que hay adentro.
    """
    os.makedirs(CARPETA_INDICES, exist_ok=True)

    ruta_vectores = os.path.join(CARPETA_INDICES, nombre + ".npy")
    ruta_metadata = os.path.join(CARPETA_INDICES, nombre + ".json")

    np.save(ruta_vectores, base["vectores"])

    with open(ruta_metadata, "w", encoding="utf-8") as archivo:
        json.dump({"modelo": base["modelo"], "registros": base["registros"]},
                  archivo, ensure_ascii=False)

    return ruta_vectores, ruta_metadata


def cargar(nombre):
    """Carga una base guardada. Devuelve None si no existe."""
    ruta_vectores = os.path.join(CARPETA_INDICES, nombre + ".npy")
    ruta_metadata = os.path.join(CARPETA_INDICES, nombre + ".json")

    if not os.path.exists(ruta_vectores):
        return None

    with open(ruta_metadata, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    return {
        "vectores": np.load(ruta_vectores),
        "registros": datos["registros"],
        "modelo": datos["modelo"],
    }


# ---------------------------------------------------------------------------
def main():
    parser = crear_parser("Una base de datos vectorial completa")
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)

    with open(ARCHIVO_DOC, "r", encoding="utf-8") as archivo:
        documento = archivo.read()

    chunks = crear_chunks_con_metadata("manual", documento, chunking_respetando_limites)

    # Le agregamos a cada chunk una categoria, deducida del encabezado de la
    # seccion en la que cae. Nos va a servir para probar el filtrado.
    seccion_actual = "general"

    for chunk in chunks:
        texto = chunk["texto"]

        if "## " in texto:
            # Nos quedamos con el texto que sigue al ultimo "## " del chunk.
            despues = texto.split("## ")[-1]
            seccion_actual = despues.split("\n")[0].strip().lower()

        chunk["seccion"] = seccion_actual

    # =======================================================================
    titulo("1) CARGAR DOCUMENTOS EN LA BASE")
    # =======================================================================

    textos = []
    for chunk in chunks:
        textos.append(chunk["texto"])

    vectores = embeber(cliente, textos, usar_cache=not args.sin_cache, mostrar=True)

    base = crear_base(cliente["modelo_embedding"])
    agregar(base, vectores, chunks)

    print()
    print("  Documentos en la base: " + str(len(base["registros"])))
    print("  Forma de la matriz   : " + str(base["vectores"].shape))
    print("  Modelo               : " + base["modelo"])
    print()

    secciones = []
    for chunk in chunks:
        if chunk["seccion"] not in secciones:
            secciones.append(chunk["seccion"])

    print("  Secciones detectadas : " + ", ".join(secciones))

    # =======================================================================
    titulo("2) BUSCAR")
    # =======================================================================

    consulta = "cuanto tarda en llegar mi pedido al interior"
    vector_consulta = embeber(cliente, [consulta], usar_cache=not args.sin_cache)[0]

    print("  Consulta: \"" + consulta + "\"")
    print()

    for resultado in buscar(base, vector_consulta, k=3):
        texto = resultado["registro"]["texto"].replace("\n", " ")

        print("  " + str(round(resultado["puntaje"], 4)) + "  [" +
              resultado["registro"]["seccion"] + "]  " + texto[:80] + "...")

    # =======================================================================
    titulo("3) BUSCAR CON FILTRO: LA FUNCION MAS SUBESTIMADA")
    # =======================================================================

    print("  Misma consulta, pero buscando SOLO en la seccion de garantias.")
    print("  (a proposito: es la seccion equivocada para esta pregunta)")
    print()

    def solo_garantias(registro):
        return registro["seccion"] == "garantias"

    resultados = buscar(base, vector_consulta, k=3, filtro=solo_garantias)

    if len(resultados) == 0:
        print("  No hay documentos en esa seccion.")
    else:
        for resultado in resultados:
            texto = resultado["registro"]["texto"].replace("\n", " ")
            print("  " + str(round(resultado["puntaje"], 4)) + "  " + texto[:80] + "...")

    print()
    print("  Fijate que devolvio algo igual, y con un puntaje que no es bajisimo.")
    print()
    print("  ESA ES LA LECCION: la busqueda vectorial SIEMPRE devuelve los k mas")
    print("  parecidos, aunque ninguno sirva. Nunca dice 'no encontre nada'.")
    print("  Decidir si el mejor resultado es lo bastante bueno es trabajo tuyo.")

    # =======================================================================
    titulo("4) LA PROTECCION QUE TE SALVA DE UN BUG INVISIBLE")
    # =======================================================================

    print("  Intentamos agregar un vector de 100 dimensiones a una base de " +
          str(base["vectores"].shape[1]) + ":")
    print()

    vector_invalido = [0.1] * 100
    registro_invalido = {"chunk_id": "falso", "texto": "no importa", "seccion": "x"}

    try:
        agregar(base, [vector_invalido], [registro_invalido])
        print("  Se agrego sin protestar. Eso seria malisimo.")
    except ValueError as error:
        print("  Rechazado, como corresponde:")
        print()
        for linea in str(error).split("\n"):
            print("    " + linea)

    print()
    print("  Sin esta verificacion, la base habria aceptado el vector y las")
    print("  busquedas empezarian a devolver resultados raros sin ningun error.")
    print("  Ese es el peor tipo de bug: el que no se queja.")

    # =======================================================================
    titulo("5) BORRAR Y PERSISTIR")
    # =======================================================================

    antes = len(base["registros"])
    id_a_borrar = base["registros"][0]["chunk_id"]

    borrar(base, id_a_borrar)

    print("  Borramos " + id_a_borrar)
    print("  Documentos: " + str(antes) + " -> " + str(len(base["registros"])))
    print("  Matriz:     " + str(base["vectores"].shape))
    print()
    print("  La matriz y la lista de metadata se achican juntas. Si se")
    print("  desincronizan, la base devuelve el texto equivocado para cada")
    print("  vector, que es un bug dificilisimo de encontrar.")

    subtitulo("Guardar en disco y volver a cargar")

    ruta_v, ruta_m = guardar(base, "manual")

    print("  Vectores : " + ruta_v)
    print("  Metadata : " + ruta_m)

    base_cargada = cargar("manual")

    print()
    print("  Recargada desde disco: " + str(len(base_cargada["registros"])) +
          " documentos, matriz " + str(base_cargada["vectores"].shape))
    print()
    print("  POR QUE IMPORTA: embeber cuesta plata y tiempo. Se hace UNA vez y")
    print("  se guarda. En produccion solo se re-embebe lo que cambio.")

    # =======================================================================
    titulo("PARA LLEVARSE")
    # =======================================================================
    print("""
  * Una base vectorial es una matriz de numeros mas una lista de metadata, en
    el mismo orden, y una multiplicacion de matrices para buscar.

  * Verifica SIEMPRE las dimensiones al agregar. Mezclar modelos corrompe la
    base en silencio.

  * El filtrado por metadata no es un extra: en produccion casi todas las
    busquedas van acotadas a un subconjunto.

  * La busqueda vectorial nunca dice "no encontre nada": siempre devuelve los
    k mas parecidos. El umbral de "esto sirve o no" lo pones vos.

  * Para menos de 50.000 documentos, esto que escribimos alcanza y sobra.
    Pasate a FAISS, pgvector o Qdrant cuando lo midas, no antes.
""")


if __name__ == "__main__":
    main()
