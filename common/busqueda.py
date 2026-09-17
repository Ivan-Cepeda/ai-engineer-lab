"""
Busqueda lexica (BM25) y fusion de rankings (RRF).

DE DONDE SALE ESTE ARCHIVO
  Del video "Las tecnicas de RAG avanzado que uso en produccion".

  El ejercicio L2-03 implementa una version SIMPLIFICADA de estas dos ideas,
  escrita paso a paso para entenderlas. Este archivo tiene la version completa:
  el BM25 de verdad, con los dos parametros que le faltaban a aquella (k1 y b).

  La parte final del ejercicio L2-03 importa este archivo y compara los dos
  rankings, para que se vea que agregan esos parametros. El concepto es el
  mismo; lo que cambia es el detalle que importa a escala.
"""

import math


def separar_en_palabras(texto):
    """
    Parte un texto en palabras: minusculas, sin signos, sin tildes.

    Sacar las tildes importa mas de lo que parece en espanol: sin eso,
    "garantia" y "garantía" son dos palabras distintas para el buscador, y el
    usuario que escribe sin tilde no encuentra nada.
    """
    tildes = [["a", "á"], ["e", "é"], ["i", "í"], ["o", "ó"], ["u", "ú"]]

    texto = texto.lower()

    for par in tildes:
        texto = texto.replace(par[1], par[0])

    limpio = ""

    for caracter in texto:
        if caracter.isalnum() or caracter == "-":
            limpio = limpio + caracter
        else:
            limpio = limpio + " "

    return limpio.split()


def preparar_indice_bm25(documentos):
    """
    Prepara todo lo que BM25 necesita saber de la coleccion, una sola vez.

    Se calcula al indexar, no en cada busqueda. Guarda:
      * cada documento partido en palabras
      * en cuantos documentos aparece cada palabra (para el IDF)
      * el largo promedio de los documentos (para normalizar)
    """
    documentos_en_palabras = []

    for texto in documentos:
        documentos_en_palabras.append(separar_en_palabras(texto))

    # En cuantos documentos aparece cada palabra.
    apariciones = {}

    for palabras in documentos_en_palabras:
        # set() saca los repetidos: nos interesa en cuantos DOCUMENTOS aparece,
        # no cuantas veces en total.
        for palabra in set(palabras):
            if palabra in apariciones:
                apariciones[palabra] = apariciones[palabra] + 1
            else:
                apariciones[palabra] = 1

    suma_de_largos = 0
    for palabras in documentos_en_palabras:
        suma_de_largos = suma_de_largos + len(palabras)

    if len(documentos_en_palabras) > 0:
        largo_promedio = suma_de_largos / len(documentos_en_palabras)
    else:
        largo_promedio = 1

    return {
        "documentos": documentos_en_palabras,
        "apariciones": apariciones,
        "total": len(documentos_en_palabras),
        "largo_promedio": largo_promedio,
    }


def buscar_bm25(indice, consulta, k=5, k1=1.5, b=0.75):
    """
    BM25: el algoritmo estandar de busqueda por palabras clave.

    Es lo que usan Elasticsearch, Lucene y practicamente cualquier buscador de
    texto desde hace 30 anios. Tiene tres partes:

    1. IDF (cuanto vale la palabra)
       Una palabra que aparece en todos los documentos no distingue nada y vale
       poco. Una que aparece en uno solo es muy informativa y vale mucho.

    2. SATURACION (el parametro k1)
       Que una palabra aparezca 20 veces no hace al documento 20 veces mas
       relevante. BM25 hace que el aporte se estanque: de 1 a 2 veces sube
       bastante, de 19 a 20 casi nada. Eso es lo que evita que un documento
       gane solo por repetir una palabra.

    3. NORMALIZACION POR LARGO (el parametro b)
       Un documento largo tiene mas chances de contener cualquier palabra por
       pura casualidad. b=0.75 penaliza a los documentos mas largos que el
       promedio.

    Los valores k1=1.5 y b=0.75 son los clasicos y casi nunca hace falta
    tocarlos.

    Devuelve una lista de (posicion, puntaje) ordenada de mejor a peor.
    """
    palabras_consulta = separar_en_palabras(consulta)

    puntajes = []

    for posicion in range(indice["total"]):
        palabras_documento = indice["documentos"][posicion]
        largo = len(palabras_documento)

        puntaje = 0.0

        for palabra in palabras_consulta:
            if palabra not in indice["apariciones"]:
                continue   # la palabra no esta en ningun documento

            veces = palabras_documento.count(palabra)

            if veces == 0:
                continue   # no esta en ESTE documento

            # --- 1. IDF ---
            documentos_con_la_palabra = indice["apariciones"][palabra]

            idf = math.log(
                1 + (indice["total"] - documentos_con_la_palabra + 0.5) /
                (documentos_con_la_palabra + 0.5)
            )

            # --- 2 y 3. Saturacion y normalizacion por largo ---
            numerador = veces * (k1 + 1)
            denominador = veces + k1 * (1 - b + b * largo / indice["largo_promedio"])

            puntaje = puntaje + idf * numerador / denominador

        puntajes.append([posicion, puntaje])

    puntajes.sort(key=lambda par: par[1], reverse=True)

    return puntajes[:k]


def fusionar_rrf(listas_de_rankings, k=60):
    """
    Reciprocal Rank Fusion: combina varios rankings en uno.

    EL PROBLEMA QUE RESUELVE
      La similitud coseno da numeros como 0.74. BM25 da numeros como 3.2. No se
      pueden sumar: estan en escalas distintas y una aplastaria a la otra.
      Normalizarlas tampoco funciona bien, porque sus distribuciones no se
      parecen en nada.

    LA SOLUCION
      Ignorar los puntajes y mirar solo la POSICION. Cada documento suma
      1 / (k + posicion) por cada lista donde aparece.

      El k=60 es un amortiguador: sin el, el #1 valdria el doble que el #2, lo
      cual es demasiado. Con 60, la diferencia entre puestos cercanos es suave.
      Ese valor sale del paper original y practicamente nadie lo cambia.

    Recibe una lista de listas de posiciones, ya ordenadas de mejor a peor.
    Devuelve una lista de (posicion, puntaje_combinado).
    """
    puntajes = {}

    for ranking in listas_de_rankings:
        for lugar in range(len(ranking)):
            documento = ranking[lugar]

            aporte = 1.0 / (k + lugar + 1)

            if documento in puntajes:
                puntajes[documento] = puntajes[documento] + aporte
            else:
                puntajes[documento] = aporte

    ordenados = sorted(puntajes.items(), key=lambda par: par[1], reverse=True)

    resultado = []
    for par in ordenados:
        resultado.append([par[0], par[1]])

    return resultado


def buscar_hibrida(indice_bm25, consulta, posiciones_semanticas, k=5, cantidad_bm25=10):
    """
    Junta la busqueda semantica con la lexica y las fusiona con RRF.

    "posiciones_semanticas" son las posiciones que ya devolvio la busqueda por
    embeddings, ordenadas de mejor a peor. Esta funcion agrega la mitad lexica
    y devuelve el ranking combinado.
    """
    resultados_bm25 = buscar_bm25(indice_bm25, consulta, k=cantidad_bm25)

    posiciones_bm25 = []
    for par in resultados_bm25:
        # Solo sumamos los que tienen puntaje real: un 0 significa que ninguna
        # palabra de la consulta aparece, y sumarlo al ranking es meter ruido.
        if par[1] > 0:
            posiciones_bm25.append(par[0])

    fusionados = fusionar_rrf([posiciones_semanticas, posiciones_bm25])

    return fusionados[:k]
