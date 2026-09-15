"""
Las cinco estrategias de fragmentacion (chunking).

NOTA SOBRE EL ORIGEN
  Las estrategias 1, 2 y 3 son las de la leccion L1.
  Las estrategias 4 (por estructura) y 5 (semantica) se agregaron despues,
  a partir del video "Las tecnicas de RAG avanzado que uso en produccion".
  Estan al final del archivo, marcadas.

QUE ES EL CHUNKING Y POR QUE HACE FALTA
  Un documento largo no se puede embeber entero. Por dos motivos:

  1. Los modelos tienen un limite de cuanto texto aceptan de una vez.
  2. Y aunque entrara, seria inutil. Un manual de 10 paginas convertido en UN
     solo vector es un promedio de todo lo que dice. Cuando busques algo, el
     sistema te va a decir "esta en el manual", que ya lo sabias. No te va a
     poder decir en que parrafo.

  Entonces partimos el documento en pedazos (chunks) que sean chicos pero que
  sigan teniendo sentido por si solos.

LOS TRES OBJETIVOS QUE COMPITEN
  * Conservar contexto: el chunk tiene que entenderse solo.
  * Precision: chunks chicos permiten senalar el parrafo exacto.
  * Eficiencia: mas chunks = mas vectores que guardar y comparar.

  No se pueden maximizar los tres. Elegir una estrategia es elegir que
  sacrificas.
"""

from common.vectores import similitud_coseno


# ---------------------------------------------------------------------------
# ESTRATEGIA 1: tamano fijo
# ---------------------------------------------------------------------------

def chunking_tamano_fijo(texto, palabras_por_chunk=80):
    """
    Corta cada N palabras, sin mirar el contenido.

    VENTAJA: es la mas simple y garantiza que ningun chunk supere el limite.
    DESVENTAJA: corta a mitad de oracion. Un chunk puede terminar en "el
                cliente tiene 30 dias para" y el siguiente empezar en
                "devolver el producto". Los dos quedan incompletos.
    CUANDO USARLA: cuando el texto no tiene estructura clara, o cuando la
                   coherencia importa poco.

    Trabajamos con palabras y no con tokens para que se pueda leer sin
    depender de una libreria. En produccion se cuenta en tokens.
    """
    palabras = texto.split()
    chunks = []

    inicio = 0

    while inicio < len(palabras):
        pedazo = palabras[inicio:inicio + palabras_por_chunk]
        chunks.append(" ".join(pedazo))
        inicio = inicio + palabras_por_chunk

    return chunks


# ---------------------------------------------------------------------------
# ESTRATEGIA 2: ventana deslizante (con solapamiento)
# ---------------------------------------------------------------------------

def chunking_ventana_deslizante(texto, palabras_por_chunk=80, solapamiento=20):
    """
    Igual que el anterior, pero cada chunk repite el final del anterior.

    LA IDEA: si una frase importante queda justo en el borde entre dos chunks,
    con solapamiento aparece completa en alguno de los dos.

    VENTAJA: casi no se pierde informacion en los cortes.
    DESVENTAJA: se repite texto. Mas chunks, mas vectores, mas plata.
    CUANTO SOLAPAR: la regla practica es entre el 10% y el 30% del tamano del
                    chunk. Mas que eso es pagar de mas por poca mejora.
    """
    if solapamiento >= palabras_por_chunk:
        raise ValueError(
            "El solapamiento tiene que ser menor que el tamano del chunk, "
            "o el bucle nunca avanza."
        )

    palabras = texto.split()
    chunks = []

    inicio = 0
    # Cuanto avanzamos en cada paso: el tamano menos lo que repetimos.
    paso = palabras_por_chunk - solapamiento

    while inicio < len(palabras):
        pedazo = palabras[inicio:inicio + palabras_por_chunk]
        chunks.append(" ".join(pedazo))

        # Si este chunk ya llego al final del texto, cortamos. Sin esto, el
        # solapamiento haria que se generen chunks repetidos al final.
        if inicio + palabras_por_chunk >= len(palabras):
            break

        inicio = inicio + paso

    return chunks


# ---------------------------------------------------------------------------
# ESTRATEGIA 3: respetando limites (recursiva / consciente de la estructura)
# ---------------------------------------------------------------------------

def dividir_en_oraciones(texto):
    """
    Parte el texto en oraciones, de forma simple.

    Buscamos punto, signo de exclamacion o de pregunta seguidos de un espacio.
    No es perfecto: "Dr. Perez" lo va a cortar mal. Las librerias serias (nltk,
    spacy) manejan esos casos, pero agregan una dependencia pesada y aca lo
    importante es entender la idea.
    """
    oraciones = []
    actual = ""

    for caracter in texto:
        actual = actual + caracter

        if caracter in ".!?":
            oraciones.append(actual.strip())
            actual = ""

    # Lo que quedo sin signo de puntuacion final tambien es una oracion.
    if actual.strip():
        oraciones.append(actual.strip())

    return oraciones


def chunking_respetando_limites(texto, palabras_por_chunk=80):
    """
    Junta oraciones completas hasta acercarse al tamano objetivo.

    Nunca corta una oracion por la mitad: si la proxima no entra, cierra el
    chunk y arranca uno nuevo.

    VENTAJA: todos los chunks se leen bien. Es la que mejor funciona en la
             practica para documentos escritos por humanos.
    DESVENTAJA: los chunks quedan de tamanos desparejos.
    CUANDO USARLA: es el mejor punto de partida para casi cualquier documento
                   con estructura: manuales, politicas, articulos, FAQs.
    """
    oraciones = dividir_en_oraciones(texto)

    chunks = []
    chunk_actual = []
    palabras_actuales = 0

    for oracion in oraciones:
        palabras_de_la_oracion = len(oracion.split())

        # Si agregar esta oracion nos pasa del limite Y ya tenemos algo
        # acumulado, cerramos el chunk y empezamos otro.
        if palabras_actuales + palabras_de_la_oracion > palabras_por_chunk and len(chunk_actual) > 0:
            chunks.append(" ".join(chunk_actual))
            chunk_actual = []
            palabras_actuales = 0

        chunk_actual.append(oracion)
        palabras_actuales = palabras_actuales + palabras_de_la_oracion

    # El ultimo chunk, que quedo sin cerrar.
    if len(chunk_actual) > 0:
        chunks.append(" ".join(chunk_actual))

    return chunks


# ---------------------------------------------------------------------------
# Chunks con metadata: lo que hace falta en un sistema real
# ---------------------------------------------------------------------------

def crear_chunks_con_metadata(documento_id, texto, funcion_de_chunking):
    """
    Aplica una estrategia de chunking y le agrega datos de trazabilidad.

    POR QUE LA METADATA NO ES OPCIONAL
      Sin ella, cuando tu sistema responde algo, no podes saber de donde lo
      saco. Con ella podes:

        * Citar la fuente: "esto sale del manual v2.3, parrafo 4".
        * Auditar: que version del documento respondio esta consulta.
        * Actualizar de a poco: re-embeber solo los documentos que cambiaron.
        * Depurar: encontrar que chunk esta arruinando los resultados.

    Recibe la funcion de chunking como parametro. Eso permite usar cualquiera
    de las tres estrategias sin duplicar este codigo.
    """
    textos = funcion_de_chunking(texto)
    chunks = []

    posicion_de_busqueda = 0

    for indice in range(len(textos)):
        texto_del_chunk = textos[indice]

        # Buscamos donde empieza este chunk dentro del documento original.
        # Arrancamos la busqueda donde termino el chunk anterior, para que dos
        # chunks con el mismo texto no devuelvan la misma posicion.
        inicio = texto.find(texto_del_chunk[:40], posicion_de_busqueda)

        if inicio == -1:
            inicio = posicion_de_busqueda

        fin = inicio + len(texto_del_chunk)
        posicion_de_busqueda = inicio + 1

        chunk = {
            "chunk_id": documento_id + "_chunk_" + str(indice).zfill(4),
            "documento_id": documento_id,
            "indice": indice,
            "texto": texto_del_chunk,
            "caracter_inicio": inicio,
            "caracter_fin": fin,
            "palabras": len(texto_del_chunk.split()),
        }

        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# ESTRATEGIA 4: chunking por estructura
#
# AGREGADA DESPUES, a partir del video "Las tecnicas de RAG avanzado que uso en
# produccion". No estaba en la primera version de este archivo.
# ---------------------------------------------------------------------------

def chunking_por_estructura(texto, palabras_maximas=250):
    """
    Corta usando los encabezados del documento, no el conteo de palabras.

    LA IDEA
      Un documento escrito por una persona YA viene dividido en ideas: eso son
      los titulos y subtitulos. Alguien se tomo el trabajo de decidir donde
      empieza y termina cada tema. Ignorarlo y cortar cada 80 palabras es tirar
      esa informacion a la basura.

    VENTAJA: cada chunk es una seccion completa y coherente. Ademas sabes a que
             seccion pertenece, lo cual sirve para filtrar y para citar.
    DESVENTAJA: depende de que el documento TENGA estructura. Con un PDF
                escaneado o un texto plano corrido, no sirve.
    EN PRODUCCION: para PDFs y documentos de Word se usan parsers como Docling,
                   que reconstruyen la jerarquia de titulos, tablas y listas.
                   Aca lo hacemos con Markdown, que ya trae los encabezados en
                   el texto.

    Si una seccion es mas larga que palabras_maximas, la partimos respetando
    oraciones, y le repetimos el titulo a cada pedazo para que no pierda el
    contexto de donde vino.
    """
    lineas = texto.split("\n")

    secciones = []
    titulo_actual = "(sin titulo)"
    lineas_actuales = []

    for linea in lineas:
        es_encabezado = linea.strip().startswith("#")

        if es_encabezado:
            # Cerramos la seccion anterior antes de arrancar la nueva.
            if len(lineas_actuales) > 0:
                cuerpo = "\n".join(lineas_actuales).strip()

                if len(cuerpo) > 0:
                    secciones.append([titulo_actual, cuerpo])

            titulo_actual = linea.strip().lstrip("#").strip()
            lineas_actuales = []
        else:
            lineas_actuales.append(linea)

    # La ultima seccion, que quedo sin cerrar.
    if len(lineas_actuales) > 0:
        cuerpo = "\n".join(lineas_actuales).strip()

        if len(cuerpo) > 0:
            secciones.append([titulo_actual, cuerpo])

    chunks = []

    for seccion in secciones:
        titulo = seccion[0]
        cuerpo = seccion[1]

        if len(cuerpo.split()) <= palabras_maximas:
            # Entra entera. Le ponemos el titulo adelante: asi el chunk sabe de
            # que seccion habla, y eso mejora bastante la busqueda.
            chunks.append(titulo + "\n" + cuerpo)
        else:
            # Es muy larga: la partimos respetando oraciones y le repetimos el
            # titulo a cada pedazo.
            for pedazo in chunking_respetando_limites(cuerpo, palabras_maximas):
                chunks.append(titulo + "\n" + pedazo)

    return chunks


# ---------------------------------------------------------------------------
# ESTRATEGIA 5: chunking semantico
#
# AGREGADA DESPUES, a partir del mismo video.
# ---------------------------------------------------------------------------

def chunking_semantico(texto, funcion_de_embedding, umbral=0.75, palabras_maximas=250):
    """
    Corta donde CAMBIA EL TEMA, medido con embeddings.

    COMO FUNCIONA
      1. Parte el texto en oraciones.
      2. Calcula el embedding de cada oracion.
      3. Compara cada oracion con la siguiente.
      4. Si se parecen (por encima del umbral), siguen en el mismo chunk.
         Si la similitud CAE, ahi cambio el tema: se abre un chunk nuevo.

    Es la unica estrategia que mira el CONTENIDO para decidir donde cortar. Las
    otras miran el largo o el formato.

    VENTAJA: cada chunk contiene una idea completa, sin cortes arbitrarios.
    DESVENTAJA: hay que embeber CADA ORACION del documento antes de poder
                indexarlo. Eso cuesta plata y tiempo en la fase de ingesta.
                Es la estrategia mas cara de las cinco.

    EL UMBRAL ES UNA PERILLA, no una constante universal:
      Mas alto  -> corta mas seguido, chunks mas chicos y mas especificos.
      Mas bajo  -> corta menos, chunks mas grandes.
      El valor correcto depende del modelo de embeddings y del documento.
      Acordate de que cada modelo tiene su propio "piso" de similitud (lo vimos
      en L1-01), asi que un umbral copiado de un tutorial casi seguro no sirve.

    "funcion_de_embedding" recibe una lista de textos y devuelve una lista de
    vectores. La pasamos como parametro para que este archivo no dependa del
    proveedor.
    """
    oraciones = dividir_en_oraciones(texto)

    # Descartamos las lineas vacias o demasiado cortas, tipicas del Markdown.
    utiles = []
    for oracion in oraciones:
        if len(oracion.split()) >= 3:
            utiles.append(oracion)

    if len(utiles) <= 1:
        return utiles

    vectores = funcion_de_embedding(utiles)

    chunks = []
    chunk_actual = [utiles[0]]

    for posicion in range(len(utiles) - 1):
        similitud = similitud_coseno(vectores[posicion], vectores[posicion + 1])

        palabras_del_chunk = len(" ".join(chunk_actual).split())

        # Cortamos por dos motivos: porque cambio el tema, o porque el chunk ya
        # se hizo demasiado grande. El segundo limite es una red de seguridad:
        # sin el, un documento muy homogeneo daria un unico chunk gigante.
        cambio_de_tema = similitud < umbral
        se_hizo_muy_largo = palabras_del_chunk >= palabras_maximas

        if cambio_de_tema or se_hizo_muy_largo:
            chunks.append(" ".join(chunk_actual))
            chunk_actual = []

        chunk_actual.append(utiles[posicion + 1])

    if len(chunk_actual) > 0:
        chunks.append(" ".join(chunk_actual))

    return chunks
