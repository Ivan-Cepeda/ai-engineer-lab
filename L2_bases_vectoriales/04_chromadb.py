"""
L2 - Ejercicio 04: ChromaDB, una base vectorial de verdad.

DE DONDE VIENE ESTE EJERCICIO
  En el ejercicio 02 escribimos una base vectorial a mano, con NumPy. Sirvio
  para entender que por dentro es una matriz y una multiplicacion.

  Ahora usamos una de verdad. Y el objetivo no es "aprender Chroma": es que
  veas que CADA COSA que escribiste a mano tiene su equivalente exacto, y que
  ya sabes lo que esta pasando por debajo.

POR QUE CHROMA Y NO OTRA
  Porque corre local, se instala con un pip y guarda todo en una carpeta. No
  necesitas levantar un servidor, ni una cuenta, ni una tarjeta. Es la que
  conviene para aprender y para prototipos.

  Pinecone, Weaviate y Qdrant son alternativas mas robustas para produccion,
  pero la API se parece muchisimo: lo que aprendas aca se traslada.

REQUISITO
    pip install chromadb

  Si no lo tenes instalado, el ejercicio te lo explica igual y no se rompe.

COMO CORRERLO
    python 04_chromadb.py
"""

import os
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.chunking import chunking_respetando_limites, crear_chunks_con_metadata
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.texto import normalizar
from common.ui import mostrar_configuracion, subtitulo, titulo


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO = os.path.join(CARPETA, "datos", "base_conocimiento.md")

# Chroma guarda todo en una carpeta. La ponemos dentro de cache/, que ya esta
# en .gitignore: una base de datos no se versiona.
CARPETA_CHROMA = os.path.join(CARPETA, "cache", "chroma")


def main():
    parser = crear_parser("ChromaDB: una base vectorial de verdad")
    parser.add_argument(
        "--reiniciar",
        action="store_true",
        help="Borrar la base y empezar de cero",
    )
    args = parser.parse_args()

    # Si no esta instalado, explicamos y salimos sin romper nada.
    try:
        import chromadb
    except ImportError:
        titulo("CHROMADB NO ESTA INSTALADO")
        print("""
  Para instalarlo:

      pip install chromadb

  Ocupa unos 100 MB con sus dependencias. Bastante menos que
  sentence-transformers, que descarga PyTorch entero.

  LO QUE TE PERDES SI NO LO CORRES
    Ver que la base vectorial que escribiste a mano en el ejercicio 02 y una
    base de produccion hacen exactamente lo mismo, con los mismos conceptos.

  Los conceptos estan explicados igual en el README de esta leccion, asi que
  podes seguir sin instalarlo.
""")
        return

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)
    print("[chromadb] version " + str(chromadb.__version__))

    if args.reiniciar and os.path.exists(CARPETA_CHROMA):
        shutil.rmtree(CARPETA_CHROMA)
        print("[chromadb] base anterior borrada")

    # =======================================================================
    titulo("1) LO QUE ESCRIBIMOS A MANO, Y SU EQUIVALENTE")
    # =======================================================================
    print("""
  Ejercicio 02 (a mano)              ChromaDB
  ---------------------------------  ------------------------------------
  crear_base(modelo)                 client.create_collection(name=...)
  base["vectores"] = matriz NumPy    lo maneja Chroma por dentro
  base["registros"] = lista          metadatas y documents
  agregar(base, vectores, registros) collection.add(...)
  buscar(base, vector, k)            collection.query(...)
  buscar(..., filtro=funcion)        collection.query(..., where={...})
  borrar(base, chunk_id)             collection.delete(ids=[...])
  guardar(base, nombre)              PersistentClient lo hace solo
  verificar dimensiones              lo verifica Chroma y lanza error

  No hay ningun concepto nuevo. Lo unico que cambia es quien escribe el
  codigo. Y eso es exactamente lo que queriamos: que una base vectorial deje
  de ser una caja negra.
""")

    # =======================================================================
    titulo("2) CREAR LA COLECCION Y CARGAR LOS DOCUMENTOS")
    # =======================================================================

    with open(ARCHIVO, "r", encoding="utf-8") as archivo:
        documento = archivo.read()

    chunks = crear_chunks_con_metadata("manual", documento, chunking_respetando_limites)

    # Le agregamos a cada chunk la seccion a la que pertenece, igual que en el
    # ejercicio 02. Nos va a servir para filtrar.
    seccion_actual = "general"

    for chunk in chunks:
        if "## " in chunk["texto"]:
            despues = chunk["texto"].split("## ")[-1]
            seccion_actual = despues.split("\n")[0].strip().lower()

        # Normalizamos el nombre: sin tildes y en minuscula.
        #
        # Esto NO es cosmetico. La primera version de este ejercicio guardaba
        # la seccion tal cual venia del markdown ("garantias" con tilde) y
        # filtraba por el texto sin tilde. El filtro no encontraba nada y no
        # habia ningun error: Chroma devolvia una lista vacia y listo.
        #
        # Los valores que usas para filtrar tienen que estar normalizados al
        # guardarlos, o vas a estar comparando contra algo que no existe.
        chunk["seccion"] = normalizar(seccion_actual)

    textos = []
    for chunk in chunks:
        textos.append(chunk["texto"])

    vectores = embeber(cliente, textos, usar_cache=not args.sin_cache, mostrar=True)

    # PersistentClient guarda en disco. Existe tambien EphemeralClient, que
    # trabaja solo en memoria y se pierde al cerrar el programa: sirve para
    # tests, no para guardar nada.
    cliente_chroma = chromadb.PersistentClient(path=CARPETA_CHROMA)

    # ---------------------------------------------------------------------
    # LA DECISION MAS IMPORTANTE DE ESTE EJERCICIO: hnsw:space
    #
    # Le dice a Chroma con que metrica comparar los vectores. Si no la pones,
    # usa "l2" (distancia euclidiana) por defecto, que NO es lo que queres
    # para embeddings de texto.
    #
    # Es el error silencioso mas comun al empezar con Chroma: los resultados
    # salen raros y no hay ningun mensaje de error que te avise.
    # ---------------------------------------------------------------------
    coleccion = cliente_chroma.get_or_create_collection(
        name="manual",
        metadata={"hnsw:space": "cosine"},
    )

    if coleccion.count() == 0:
        # Chroma necesita tres listas paralelas, en el mismo orden:
        #   ids       -> identificador unico de cada documento
        #   embeddings -> los vectores (le pasamos los NUESTROS, ver abajo)
        #   documents -> el texto, para poder devolverlo en la busqueda
        #   metadatas -> los datos extra, para filtrar
        identificadores = []
        metadatos = []

        for chunk in chunks:
            identificadores.append(chunk["chunk_id"])
            metadatos.append({
                "seccion": chunk["seccion"],
                "indice": chunk["indice"],
                "palabras": chunk["palabras"],
            })

        coleccion.add(
            ids=identificadores,
            embeddings=vectores,
            documents=textos,
            metadatas=metadatos,
        )

        print()
        print("  Cargados " + str(len(chunks)) + " documentos en la coleccion.")
    else:
        print()
        print("  La coleccion ya tenia " + str(coleccion.count()) + " documentos.")
        print("  Corre con --reiniciar si queres empezar de cero.")

    print("  Guardada en: " + CARPETA_CHROMA)

    subtitulo("Por que le pasamos NUESTROS embeddings")
    print("""
  Chroma puede generar los embeddings solo: si no le pasas el parametro
  embeddings, usa un modelo local propio (all-MiniLM-L6-v2, el mismo de L4).

  Nosotros se los pasamos a proposito, por dos razones:

  1. Para seguir usando el proveedor configurado en el .env. Todo el modulo se
     apoya en que el mismo codigo funcione con OpenAI o con Gemini, y dejar
     que Chroma elija por su cuenta romperia eso.

  2. Para reutilizar el cache. Los vectores ya estaban calculados de los
     ejercicios anteriores: cargar la base no costo ni una llamada a la API.

  La contra: sos responsable de embeber la consulta con EL MISMO modelo. Si te
  equivocas, Chroma no puede darse cuenta.
""")

    # =======================================================================
    titulo("3) BUSCAR, Y EL DETALLE QUE CONFUNDE A TODOS")
    # =======================================================================

    consulta = "cuanto tarda en llegar un pedido al interior"
    vector_consulta = embeber(cliente, [consulta], usar_cache=not args.sin_cache)[0]

    print("  Consulta: \"" + consulta + "\"")
    print()

    inicio = time.time()

    resultados = coleccion.query(
        query_embeddings=[vector_consulta],
        n_results=3,
    )

    tiempo = time.time() - inicio

    # La respuesta viene envuelta en una lista por cada consulta, porque query()
    # acepta varias consultas a la vez. Como mandamos una sola, agarramos la
    # posicion 0 de cada campo.
    for lugar in range(len(resultados["ids"][0])):
        identificador = resultados["ids"][0][lugar]
        distancia = resultados["distances"][0][lugar]
        texto = resultados["documents"][0][lugar].replace("\n", " ")
        seccion = resultados["metadatas"][0][lugar]["seccion"]

        print("  #" + str(lugar + 1) + "  distancia " + str(round(distancia, 4)) +
              "  [" + seccion + "]")
        print("      " + texto[:85] + "...")

    print()
    print("  Busqueda en " + str(round(tiempo * 1000, 1)) + " ms")

    subtitulo("OJO: Chroma devuelve DISTANCIAS, no similitudes")
    print("""
  Nuestra base del ejercicio 02 devuelve similitud coseno: MAS ALTO es MEJOR,
  y va de -1 a 1.

  Chroma devuelve distancia coseno: MAS BAJO es MEJOR, y va de 0 a 2.

  La relacion es directa:

      distancia = 1 - similitud

  Si ordenas los resultados de Chroma de mayor a menor creyendo que son
  similitudes, te quedas con los PEORES resultados y no hay ningun error que
  te lo avise. Es el mismo problema que vimos en L1-02 con la distancia
  euclidiana.""")

    print()
    print("  Comprobacion sobre el primer resultado:")

    distancia = resultados["distances"][0][0]
    print("    distancia que devolvio Chroma : " + str(round(distancia, 4)))
    print("    similitud equivalente (1 - d) : " + str(round(1 - distancia, 4)))

    # =======================================================================
    titulo("4) FILTRAR POR METADATA")
    # =======================================================================

    print("  Misma consulta, pero solo en la seccion de garantias.")
    print("  (a proposito: es la seccion equivocada para esta pregunta)")
    print()

    resultados_filtrados = coleccion.query(
        query_embeddings=[vector_consulta],
        n_results=2,
        where={"seccion": "garantias"},
    )

    cantidad = len(resultados_filtrados["ids"][0])

    if cantidad == 0:
        # Si el filtro no encuentra nada, Chroma no protesta: devuelve una
        # lista vacia y listo. Casi siempre significa que el valor que estas
        # buscando no se escribio igual al guardarlo.
        print("  No hay documentos con esa seccion.")
        print()
        print("  Chroma compara el filtro como texto EXACTO. Si esperabas")
        print("  resultados, revisa como quedo escrito el valor al guardarlo:")
        print("  una tilde de diferencia alcanza para no encontrar nada.")
    else:
        for lugar in range(cantidad):
            distancia = resultados_filtrados["distances"][0][lugar]
            texto = resultados_filtrados["documents"][0][lugar].replace("\n", " ")

            print("  distancia " + str(round(distancia, 4)) + "  " + texto[:75] + "...")

        print()
        print("  Fijate que devolvio resultados igual, con una distancia que no")
        print("  es enorme, aunque la seccion no tenga nada que ver con lo que")
        print("  preguntamos.")
        print()
        print("  LA MISMA LECCION DEL EJERCICIO 02: la busqueda vectorial SIEMPRE")
        print("  devuelve los k mas parecidos de lo que quede despues del filtro,")
        print("  aunque ninguno sirva. Nunca dice 'no encontre nada'. Poner el")
        print("  umbral es trabajo tuyo, lo escribas vos o uses Chroma.")

    subtitulo("Filtros mas complejos")
    print("  Chroma acepta operadores, no solo igualdad:")
    print()

    largos = coleccion.query(
        query_embeddings=[vector_consulta],
        n_results=3,
        where={"palabras": {"$gt": 70}},
    )

    print("  where={\"palabras\": {\"$gt\": 70}}  -> solo fragmentos de mas de 70 palabras")
    print("  Devolvio " + str(len(largos["ids"][0])) + " resultados:")

    for lugar in range(len(largos["ids"][0])):
        palabras = largos["metadatas"][0][lugar]["palabras"]
        print("    " + str(palabras) + " palabras  " +
              largos["documents"][0][lugar].replace("\n", " ")[:60] + "...")

    # =======================================================================
    titulo("5) LA PROTECCION DE DIMENSIONES, QUE TAMBIEN ESTA")
    # =======================================================================

    print("  En el ejercicio 02 escribimos a mano una verificacion para que")
    print("  mezclar modelos fallara ruidosamente. Chroma la trae incorporada:")
    print()

    try:
        coleccion.add(
            ids=["vector_invalido"],
            embeddings=[[0.1] * 100],
            documents=["no importa"],
            metadatas=[{"seccion": "x", "indice": 999, "palabras": 3}],
        )
        print("  Lo acepto sin protestar. Eso seria malisimo.")
    except Exception as error:
        mensaje = str(error).split("\n")[0]
        print("  Rechazado, como corresponde:")
        print("    " + mensaje[:150])

    print()
    print("  Que una base de produccion traiga esta verificacion confirma que")
    print("  no era una paranoia nuestra: mezclar embeddings de modelos")
    print("  distintos es un error real y frecuente.")

    # =======================================================================
    titulo("6) PERSISTENCIA, ACTUALIZAR Y BORRAR")
    # =======================================================================

    antes = coleccion.count()

    id_a_borrar = chunks[0]["chunk_id"]
    coleccion.delete(ids=[id_a_borrar])

    print("  Borramos " + id_a_borrar)
    print("  Documentos: " + str(antes) + " -> " + str(coleccion.count()))

    # upsert: si el id existe lo actualiza, si no lo crea. Es lo que usarias
    # al re-indexar un documento que cambio.
    coleccion.upsert(
        ids=[id_a_borrar],
        embeddings=[vectores[0]],
        documents=[textos[0]],
        metadatas=[{"seccion": chunks[0]["seccion"], "indice": 0,
                    "palabras": chunks[0]["palabras"]}],
    )

    print("  Lo devolvimos con upsert. Documentos: " + str(coleccion.count()))
    print()
    print("  UPSERT es lo que usarias al re-indexar: si el documento ya estaba")
    print("  lo reemplaza, y si no lo crea. Sin eso tendrias que preguntar")
    print("  primero si existe, que son dos operaciones en vez de una.")

    subtitulo("La persistencia es automatica")

    # Abrimos un cliente NUEVO apuntando a la misma carpeta, como si el
    # programa se hubiera cerrado y vuelto a abrir.
    otro_cliente = chromadb.PersistentClient(path=CARPETA_CHROMA)
    misma_coleccion = otro_cliente.get_collection(name="manual")

    print("  Abrimos un cliente nuevo sobre la misma carpeta:")
    print("    documentos encontrados: " + str(misma_coleccion.count()))
    print()
    print("  No hubo que llamar a ningun guardar(). Chroma escribe a disco a")
    print("  medida que trabajas. En nuestra base del ejercicio 02 teniamos que")
    print("  acordarnos de llamar a guardar() nosotros.")

    # =======================================================================
    titulo("CUANDO USAR CADA COSA")
    # =======================================================================
    print("""
  LA QUE ESCRIBIMOS A MANO (ejercicio 02)
    A favor: cero dependencias, la entendes entera, y para menos de 50.000
             documentos anda perfecto.
    En contra: la mantenes vos.

  CHROMA
    A favor: persistencia automatica, filtros con operadores, indices
             aproximados cuando la coleccion crece, y una API que se parece a
             la de las bases de produccion.
    En contra: una dependencia mas, y tiene su propia forma de hacer las
               cosas que hay que aprender.

  PINECONE, WEAVIATE, QDRANT
    Para cuando necesites que corra en un servidor, escale a millones de
    vectores y varias aplicaciones consulten la misma base.

  EL CONSEJO PRACTICO
    Empeza con lo mas simple que funcione. Si tenes 500 documentos, NumPy
    sobra y montar infraestructura es trabajo que no te pidieron.

    Pasate cuando te lo pida un numero medido, no cuando te lo pida la
    intuicion o un blog.

  LO QUE DE VERDAD IMPORTA
    Despues del ejercicio 02 y de este, ya sabes que hay debajo de cualquiera
    de estas herramientas: una matriz de vectores, una metrica de distancia y
    una busqueda de los mas cercanos. Cuando leas la documentacion de
    cualquier base vectorial, vas a reconocer cada concepto.
""")


if __name__ == "__main__":
    main()
