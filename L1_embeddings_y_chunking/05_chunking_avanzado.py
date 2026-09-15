"""
L1 - Ejercicio 05: chunking semantico y por estructura.

=============================================================================
SALVEDAD: DE DONDE SALE ESTE EJERCICIO
  Las tres estrategias del ejercicio 03 (tamano fijo, ventana deslizante y
  respetando limites) son las de la leccion. Estas dos se agregaron despues,
  a partir del video "Las tecnicas de RAG avanzado que uso en produccion".

  El video sostiene que son las dos que conviene usar en produccion, y que
  cortar por tamano fijo es la primera causa de que un RAG falle.
=============================================================================

LAS DOS ESTRATEGIAS NUEVAS

  POR ESTRUCTURA
    Usa los titulos del documento para cortar. Un documento escrito por una
    persona YA viene dividido en ideas: alguien decidio donde empieza y termina
    cada tema. Cortar cada 80 palabras es tirar esa informacion a la basura.

  SEMANTICA
    Corta donde CAMBIA EL TEMA, medido con embeddings: compara cada oracion
    con la siguiente y abre un chunk nuevo cuando la similitud cae.
    Es la unica estrategia que mira el CONTENIDO para decidir donde cortar.

COMO CORRERLO
    python 05_chunking_avanzado.py
    python 05_chunking_avanzado.py --umbral 0.85
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.chunking import (chunking_por_estructura, chunking_respetando_limites,
                             chunking_semantico, chunking_tamano_fijo)
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.texto import contiene
from common.ui import mostrar_configuracion, mostrar_uso_del_cache, subtitulo, titulo
from common.vectores import a_matriz, buscar_mas_similares


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO = os.path.join(CARPETA, "datos", "base_conocimiento.md")


# Consultas con el texto que tiene que traer el fragmento correcto.
CASOS = [
    ["cuantos dias tengo para devolver un producto", "30 dias corridos"],
    ["cuanto sale el envio al interior", "26 en el interior"],
    ["que garantia tienen los notebooks", "24 meses"],
    ["no me llega el mail de recuperacion", "carpeta de spam"],
    ["puedo pagar con criptomonedas", "aceptamos criptomonedas"],
    ["cuanto tarda el reembolso si pague en cuotas", "dos ciclos de facturacion"],
]


def evaluar(cliente, nombre, chunks, usar_cache):
    """
    Mide que tan bien recupera esta particion del documento.

    Devuelve cantidad de chunks, aciertos en el top-3 y MRR.
    """
    vectores = embeber(cliente, chunks, usar_cache=usar_cache)
    matriz = a_matriz(vectores)

    consultas = []
    for caso in CASOS:
        consultas.append(caso[0])

    vectores_consulta = embeber(cliente, consultas, usar_cache=usar_cache)

    aciertos = 0
    suma_reciproca = 0.0

    for numero in range(len(CASOS)):
        esperado = CASOS[numero][1]

        resultados = buscar_mas_similares(vectores_consulta[numero], matriz,
                                          k=len(chunks))

        puesto = None

        for lugar in range(len(resultados)):
            if contiene(chunks[resultados[lugar][0]], esperado):
                puesto = lugar + 1
                break

        if puesto is not None:
            suma_reciproca = suma_reciproca + 1.0 / puesto

            if puesto <= 3:
                aciertos = aciertos + 1

    tamanos = []
    for chunk in chunks:
        tamanos.append(len(chunk.split()))

    return {
        "nombre": nombre,
        "chunks": len(chunks),
        "aciertos": aciertos,
        "mrr": round(suma_reciproca / len(CASOS), 3),
        "min": min(tamanos),
        "max": max(tamanos),
        "promedio": round(sum(tamanos) / len(tamanos)),
    }


def main():
    parser = crear_parser("Chunking semantico y por estructura")
    parser.add_argument(
        "--umbral",
        type=float,
        default=0.75,
        help="Umbral de similitud del chunking semantico: mas alto corta mas",
    )
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)
    usar_cache = not args.sin_cache

    with open(ARCHIVO, "r", encoding="utf-8") as archivo:
        documento = archivo.read()

    # =======================================================================
    titulo("1) CHUNKING POR ESTRUCTURA")
    # =======================================================================

    estructurales = chunking_por_estructura(documento)

    print("  Genero " + str(len(estructurales)) + " chunks, uno por seccion del manual.")
    print()

    for indice in range(min(3, len(estructurales))):
        primera_linea = estructurales[indice].split("\n")[0]
        palabras = len(estructurales[indice].split())

        print("  --- chunk " + str(indice) + " (" + str(palabras) + " palabras) ---")
        print("  Titulo: " + primera_linea)
        print()

    print("  FIJATE EN EL DETALLE QUE MAS RINDE:")
    print("  A cada chunk le pegamos adelante el titulo de su seccion. Asi el")
    print("  fragmento sabe de que tema habla, aunque el texto no lo repita.")
    print("  Un parrafo que dice 'el plazo es de 10 dias habiles' no sirve de")
    print("  nada suelto; con 'Reembolsos' adelante, si.")
    print()
    print("  EN PRODUCCION: con PDFs y documentos de Word se usan parsers como")
    print("  Docling, que reconstruyen la jerarquia de titulos, tablas y listas.")
    print("  Aca lo hacemos con Markdown, que ya trae los encabezados en el texto.")

    # =======================================================================
    titulo("2) CHUNKING SEMANTICO")
    # =======================================================================

    print("  Compara cada oracion con la siguiente. Si la similitud CAE por")
    print("  debajo del umbral (" + str(args.umbral) + "), ahi cambio el tema y")
    print("  se abre un chunk nuevo.")
    print()

    # La funcion de chunking semantico necesita embeber, pero no tiene por que
    # saber de que proveedor. Le pasamos una funcion que ya sabe hacerlo.
    def embeber_oraciones(lista):
        return embeber(cliente, lista, usar_cache=usar_cache)

    inicio = time.time()
    semanticos = chunking_semantico(documento, embeber_oraciones, umbral=args.umbral)
    tiempo_semantico = time.time() - inicio

    print("  Genero " + str(len(semanticos)) + " chunks en " +
          str(round(tiempo_semantico, 2)) + " s")
    print()

    for indice in range(min(3, len(semanticos))):
        texto = semanticos[indice].replace("\n", " ")
        palabras = len(semanticos[indice].split())

        print("  --- chunk " + str(indice) + " (" + str(palabras) + " palabras) ---")
        print("  " + texto[:140] + "...")
        print()

    print("  EL COSTO: para decidir donde cortar hubo que embeber CADA ORACION")
    print("  del documento. Es la estrategia mas cara de las cinco, y se paga")
    print("  en la ingesta, no en la busqueda.")
    print()
    print("  EL UMBRAL ES UNA PERILLA, no una constante. Probalo:")
    print("     python 05_chunking_avanzado.py --umbral 0.85")
    print("  Mas alto corta mas seguido y da chunks mas chicos.")

    # =======================================================================
    titulo("3) LAS CINCO ESTRATEGIAS, MEDIDAS")
    # =======================================================================

    print("  Misma pregunta a las cinco particiones del mismo documento.")
    print()

    mediciones = [
        evaluar(cliente, "tamano fijo",
                chunking_tamano_fijo(documento, 80), usar_cache),
        evaluar(cliente, "respetando limites",
                chunking_respetando_limites(documento, 80), usar_cache),
        evaluar(cliente, "por estructura", estructurales, usar_cache),
        evaluar(cliente, "semantico " + str(args.umbral), semanticos, usar_cache),
    ]

    print("  Estrategia            Chunks   Top-3     MRR    min   max   prom")
    print("  " + "-" * 66)

    for m in mediciones:
        linea = "  "
        linea = linea + m["nombre"].ljust(22)
        linea = linea + str(m["chunks"]).rjust(6)
        linea = linea + (str(m["aciertos"]) + "/" + str(len(CASOS))).rjust(8)
        linea = linea + str(m["mrr"]).rjust(8)
        linea = linea + str(m["min"]).rjust(7)
        linea = linea + str(m["max"]).rjust(6)
        linea = linea + str(m["promedio"]).rjust(7)
        print(linea)

    print()
    mostrar_uso_del_cache(cliente)

    # Buscamos el mejor MRR y TODAS las que lo empatan. Reportar un solo
    # ganador cuando hay empate es enganioso: da la impresion de que una
    # estrategia es superior cuando en realidad no se distinguen.
    mejor_mrr = 0.0
    for m in mediciones:
        if m["mrr"] > mejor_mrr:
            mejor_mrr = m["mrr"]

    ganadoras = []
    for m in mediciones:
        if m["mrr"] == mejor_mrr:
            ganadoras.append(m["nombre"])

    print()

    if len(ganadoras) == 1:
        print("  En ESTE documento y con ESTAS 6 consultas gano: " +
              ganadoras[0] + " (MRR " + str(mejor_mrr) + ")")
    else:
        print("  EMPATE en MRR " + str(mejor_mrr) + " entre " +
              str(len(ganadoras)) + " estrategias:")
        print("    " + ", ".join(ganadoras))
        print()
        print("  Y ACA VIENE LO INTERESANTE.")
        print()
        print("  Si 'tamano fijo' esta entre las ganadoras, acabas de ver algo")
        print("  que contradice lo que dicen casi todos los tutoriales (y el")
        print("  video del que salio este ejercicio): que cortar por tamano fijo")
        print("  es la causa principal de que un RAG falle.")
        print()
        print("  No es que el video se equivoque. Es que nuestro manual tiene")
        print("  13 fragmentos y 6 consultas faciles. Con tan poco, la busqueda")
        print("  acierta igual aunque cortes mal: no hay ruido contra el cual")
        print("  competir.")
        print()
        print("  Las estrategias avanzadas se inventaron para bases de miles de")
        print("  documentos, donde el fragmento correcto pelea contra cientos")
        print("  de fragmentos parecidos. Ahi si se nota la diferencia.")

    print()
    print("  LA LECCION, QUE VALE MAS QUE CUALQUIER TABLA:")
    print("  Con 6 consultas, una diferencia chica no significa nada, y un")
    print("  empate tampoco demuestra que las estrategias sean equivalentes.")
    print("  No copies esta conclusion: repeti la medicion con TUS documentos.")

    # =======================================================================
    titulo("COMO ELEGIR, CON LAS CINCO SOBRE LA MESA")
    # =======================================================================
    print("""
  Estrategia           Mira        Costo ingesta   Cuando usarla
  -------------------  ----------  --------------  ----------------------------
  Tamano fijo          el largo    cero            Casi nunca. Es la linea base
  Ventana deslizante   el largo    cero            Si se pierden datos en los
                                                   bordes
  Respetando limites   la puntua-  cero            Buen punto de partida para
                       cion                        texto corrido
  Por estructura       los         cero            Documentos CON titulos:
                       titulos                     manuales, politicas, wikis
  Semantico            el signi-   alto: hay que   Texto largo sin estructura,
                       ficado      embeber cada    donde los temas cambian sin
                                   oracion         aviso

  LA RECOMENDACION DEL VIDEO, Y COINCIDO
    Empeza por ESTRUCTURA si tu documento tiene titulos. Es gratis, respeta la
    division que penso el autor, y te regala metadata (de que seccion viene
    cada chunk) que despues sirve para filtrar y para citar.

    Pasate a SEMANTICO solo si tus documentos no tienen estructura o si al
    medir ves que la estructura no alcanza. Es caro y no siempre gana.

  LO QUE NO CAMBIA
    Ninguna de las cinco es "la correcta". La correcta es la que gane con TUS
    documentos y TUS consultas, medida como en la tabla de arriba.
""")


if __name__ == "__main__":
    main()
