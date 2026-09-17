"""
L2 - Ejercicio 03: busqueda hibrida (semantica + palabras clave).

EL PROBLEMA QUE RESUELVE
  La busqueda semantica es buenisima entendiendo intencion: "no me anda
  internet" encuentra el articulo sobre WiFi aunque no compartan palabras.

  Pero es sorprendentemente mala con datos exactos. Si el cliente busca el
  codigo "SKU-200" o la palabra "express", la busqueda semantica te puede
  traer cualquier cosa parecida, porque para el modelo "SKU-200" y "SKU-300"
  significan casi lo mismo.

  La busqueda por palabras clave es exactamente al reves: perfecta con datos
  exactos, inutil con sinonimos.

LA SOLUCION
  Usar las dos y combinar los resultados. Eso es la busqueda hibrida, y es lo
  que usan casi todos los sistemas RAG serios.

COMO CORRERLO
    python 03_busqueda_hibrida.py
"""

import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.busqueda import buscar_bm25, preparar_indice_bm25
from common.chunking import chunking_respetando_limites, crear_chunks_con_metadata
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.ui import mostrar_configuracion, subtitulo, titulo
from common.vectores import a_matriz, buscar_mas_similares


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO = os.path.join(CARPETA, "datos", "base_conocimiento.md")


# ---------------------------------------------------------------------------
# La parte de palabras clave: BM25 simplificado
# ---------------------------------------------------------------------------

def separar_en_palabras(texto):
    """
    Parte un texto en palabras, en minuscula y sin signos de puntuacion.

    A esto se le llama tokenizar. Es deliberadamente simple: en produccion se
    le suma quitar tildes, reducir palabras a su raiz (stemming) y descartar
    palabras vacias como "de" o "la".
    """
    limpio = ""

    for caracter in texto.lower():
        if caracter.isalnum() or caracter == "-":
            limpio = limpio + caracter
        else:
            limpio = limpio + " "

    return limpio.split()


def puntaje_por_palabras(consulta, documentos_en_palabras):
    """
    Calcula, para cada documento, que tan bien coincide con las palabras
    de la consulta. Es una version simplificada de BM25.

    LA IDEA CENTRAL: no todas las palabras valen lo mismo.

      * Una palabra que aparece en TODOS los documentos (como "el" o
        "cliente") casi no aporta informacion: si buscas por ella, todo
        coincide. Vale poco.

      * Una palabra que aparece en UN solo documento (como "express" o
        "SKU-200") es muy informativa: si coincide, probablemente sea ESE el
        documento que buscas. Vale mucho.

    Ese peso se llama IDF: frecuencia inversa en los documentos. Se calcula
    con un logaritmo para que la diferencia no se dispare.
    """
    palabras_consulta = separar_en_palabras(consulta)
    total_documentos = len(documentos_en_palabras)

    puntajes = []

    for palabras_del_documento in documentos_en_palabras:
        puntaje = 0.0

        for palabra in palabras_consulta:
            # En cuantos documentos aparece esta palabra?
            documentos_con_la_palabra = 0

            for otras_palabras in documentos_en_palabras:
                if palabra in otras_palabras:
                    documentos_con_la_palabra = documentos_con_la_palabra + 1

            if documentos_con_la_palabra == 0:
                continue   # la palabra no esta en ningun lado

            # IDF: cuanto mas rara la palabra, mas alto el peso.
            idf = math.log(1 + total_documentos / documentos_con_la_palabra)

            # Cuantas veces aparece en ESTE documento.
            veces = palabras_del_documento.count(palabra)

            if veces > 0:
                # Dividimos por el largo del documento para que un documento
                # largo no gane solo por tener mas palabras.
                frecuencia = veces / len(palabras_del_documento)
                puntaje = puntaje + idf * frecuencia

        puntajes.append(puntaje)

    return puntajes


# ---------------------------------------------------------------------------
# La combinacion: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def fusionar_por_posicion(lista_a, lista_b, k=60):
    """
    Combina dos rankings usando Reciprocal Rank Fusion (RRF).

    EL PROBLEMA QUE RESUELVE
      La similitud coseno da numeros como 0.74. El puntaje de palabras clave
      da numeros como 0.031. No se pueden sumar: estan en escalas distintas y
      una aplastaria a la otra.

    LA SOLUCION DE RRF
      Ignorar los puntajes y mirar solo la POSICION en cada ranking. Cada
      documento suma 1 / (k + posicion) por cada lista donde aparece.

      Un documento que sale #1 en las dos listas suma mucho. Uno que sale #1
      en una y no aparece en la otra suma bastante. Uno que sale #8 en las dos
      suma poco.

      El k=60 es un amortiguador: evita que el #1 sea desproporcionadamente
      mejor que el #2. Ese 60 es el valor del paper original y se usa casi
      siempre tal cual.

    Recibe dos listas de posiciones ya ordenadas de mejor a peor.
    """
    puntajes = {}

    for lista in [lista_a, lista_b]:
        for posicion_en_el_ranking in range(len(lista)):
            documento = lista[posicion_en_el_ranking]

            aporte = 1.0 / (k + posicion_en_el_ranking + 1)

            if documento in puntajes:
                puntajes[documento] = puntajes[documento] + aporte
            else:
                puntajes[documento] = aporte

    # Ordenamos de mayor a menor puntaje combinado.
    ordenados = sorted(puntajes.items(), key=lambda par: par[1], reverse=True)

    return ordenados


def main():
    parser = crear_parser("Busqueda hibrida: semantica mas palabras clave")
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)

    with open(ARCHIVO, "r", encoding="utf-8") as archivo:
        documento = archivo.read()

    chunks = crear_chunks_con_metadata("manual", documento, chunking_respetando_limites)

    textos = []
    for chunk in chunks:
        textos.append(chunk["texto"])

    vectores = embeber(cliente, textos, usar_cache=not args.sin_cache, mostrar=True)
    matriz = a_matriz(vectores)

    # Preparamos la parte de palabras: cada documento como lista de palabras.
    documentos_en_palabras = []
    for texto in textos:
        documentos_en_palabras.append(separar_en_palabras(texto))

    # Dos consultas elegidas para que se vea la diferencia:
    #   la primera favorece a la busqueda semantica
    #   la segunda favorece a la busqueda por palabras
    consultas = [
        "no me anda internet en casa",
        "envio express",
    ]

    for consulta in consultas:

        titulo("CONSULTA: \"" + consulta + "\"")

        vector_consulta = embeber(cliente, [consulta], usar_cache=not args.sin_cache)[0]

        # --- Busqueda semantica ---
        resultados_semanticos = buscar_mas_similares(vector_consulta, matriz, k=5)

        ranking_semantico = []
        for par in resultados_semanticos:
            ranking_semantico.append(par[0])

        subtitulo("A) Solo busqueda semantica (embeddings)")

        for lugar in range(3):
            indice = resultados_semanticos[lugar][0]
            puntaje = resultados_semanticos[lugar][1]
            texto = textos[indice].replace("\n", " ")

            print("  #" + str(lugar + 1) + "  " + str(round(puntaje, 4)) +
                  "  chunk " + str(indice) + "  " + texto[:75] + "...")

        # --- Busqueda por palabras clave ---
        puntajes_palabras = puntaje_por_palabras(consulta, documentos_en_palabras)

        pares = []
        for indice in range(len(puntajes_palabras)):
            pares.append([indice, puntajes_palabras[indice]])

        pares.sort(key=lambda par: par[1], reverse=True)

        ranking_palabras = []
        for par in pares[:5]:
            ranking_palabras.append(par[0])

        subtitulo("B) Solo busqueda por palabras clave")

        for lugar in range(3):
            indice = pares[lugar][0]
            puntaje = pares[lugar][1]
            texto = textos[indice].replace("\n", " ")

            if puntaje == 0:
                print("  #" + str(lugar + 1) + "  0.0000  (ninguna palabra de la " +
                      "consulta aparece en este documento)")
            else:
                print("  #" + str(lugar + 1) + "  " + str(round(puntaje, 4)) +
                      "  chunk " + str(indice) + "  " + texto[:75] + "...")

        # --- Fusion ---
        subtitulo("C) Busqueda hibrida (las dos, fusionadas con RRF)")

        fusionados = fusionar_por_posicion(ranking_semantico, ranking_palabras)

        for lugar in range(min(3, len(fusionados))):
            indice = fusionados[lugar][0]
            puntaje = fusionados[lugar][1]
            texto = textos[indice].replace("\n", " ")

            # Marcamos de donde venia cada resultado.
            origen = []
            if indice in ranking_semantico:
                origen.append("semantica #" + str(ranking_semantico.index(indice) + 1))
            if indice in ranking_palabras:
                origen.append("palabras #" + str(ranking_palabras.index(indice) + 1))

            print("  #" + str(lugar + 1) + "  " + str(round(puntaje, 4)) +
                  "  chunk " + str(indice) + "  [" + ", ".join(origen) + "]")
            print("      " + texto[:75] + "...")

    # =======================================================================
    titulo("LO QUE LE FALTA A NUESTRO BM25")
    # =======================================================================
    # La funcion puntaje_por_palabras() que escribimos arriba es una version
    # simplificada. El BM25 de verdad, el que usan Elasticsearch y Lucene desde
    # hace 30 anios, agrega dos cosas que aca faltan.

    print("  Nuestro puntaje_por_palabras() tiene el IDF (que las palabras raras")
    print("  valgan mas), pero le faltan dos piezas del BM25 real:")
    print()
    print("  1. SATURACION (el parametro k1)")
    print("     Que una palabra aparezca 20 veces no hace al documento 20 veces")
    print("     mas relevante. BM25 hace que el aporte se estanque: de 1 a 2")
    print("     veces sube bastante, de 19 a 20 casi nada. Sin eso, un documento")
    print("     puede ganar solo por repetir una palabra.")
    print()
    print("  2. NORMALIZACION POR LARGO (el parametro b)")
    print("     Un documento largo contiene cualquier palabra por pura")
    print("     casualidad. BM25 penaliza a los mas largos que el promedio.")
    print()
    print("  La version completa esta en common/busqueda.py. Comparemos:")

    indice_completo = preparar_indice_bm25(textos)

    for consulta in consultas:
        subtitulo("\"" + consulta + "\"")

        # --- nuestra version simplificada ---
        puntajes_simple = puntaje_por_palabras(consulta, documentos_en_palabras)

        pares_simple = []
        for indice in range(len(puntajes_simple)):
            pares_simple.append([indice, puntajes_simple[indice]])

        pares_simple.sort(key=lambda par: par[1], reverse=True)

        # --- la version completa ---
        resultados_completo = buscar_bm25(indice_completo, consulta, k=3)

        print("  Simplificada          BM25 completo")
        print("  " + "-" * 44)

        for lugar in range(3):
            indice_simple = pares_simple[lugar][0]
            indice_completo_pos = resultados_completo[lugar][0]

            izquierda = "chunk " + str(indice_simple) + " (" + str(round(pares_simple[lugar][1], 3)) + ")"
            derecha = "chunk " + str(indice_completo_pos) + " (" + str(round(resultados_completo[lugar][1], 3)) + ")"

            if indice_simple == indice_completo_pos:
                marca = "  ="
            else:
                marca = "  <- distinto"

            print("  " + izquierda.ljust(22) + derecha + marca)

    print()
    print("  Si los rankings coinciden, es porque nuestro documento tiene")
    print("  fragmentos de tamanos parecidos y pocas palabras repetidas: los dos")
    print("  ajustes de BM25 no tienen nada que corregir.")
    print()
    print("  En una base real, con documentos de 50 y de 5000 palabras mezclados,")
    print("  la diferencia se vuelve grande. Por eso se usa el completo.")

    # =======================================================================
    titulo("QUE MIRAR EN LOS RESULTADOS")
    # =======================================================================
    print("""
  EN LA PRIMERA CONSULTA ("no me anda internet en casa")
    La busqueda por palabras clave tiene un problema serio: la palabra
    "internet" quizas ni aparezca en el manual, que habla de "red WiFi". Si
    ninguna palabra coincide, el puntaje es 0 y el ranking es basura.
    La semantica, en cambio, entiende de que se trata.

  EN LA SEGUNDA CONSULTA ("envio express")
    Aca las palabras clave brillan: "express" aparece en un solo chunk y es
    una palabra rara, asi que el IDF la premia fuerte. La semantica puede
    confundirse con otros chunks que hablan de envios en general.

  LA FUSION
    Un documento que aparece bien posicionado en las DOS listas sube al tope.
    Eso es exactamente lo que queres: consenso entre dos formas distintas de
    buscar es una senal mucho mas fuerte que ser bueno en una sola.

  CUANDO USAR HIBRIDA
    Casi siempre, si tu dominio tiene datos exactos: codigos de producto,
    numeros de version, nombres propios, siglas, precios. O sea: casi todos
    los dominios reales.

    Es la diferencia mas grande entre un RAG de demo y uno que la gente usa.
""")


if __name__ == "__main__":
    main()
