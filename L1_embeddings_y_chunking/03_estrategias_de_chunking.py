"""
L1 - Ejercicio 03: las tres estrategias de chunking.

EL PROBLEMA
  Un documento largo no se puede embeber entero. Y aunque entrara, no
  conviene: un manual de 10 paginas convertido en UN vector es el promedio de
  todo lo que dice. Cuando busques algo, el sistema te va a decir "esta en el
  manual". Ya lo sabias.

  Hay que partirlo en pedazos. A eso se le llama chunking. Como lo partas
  determina la calidad de todo lo que venga despues.

LO QUE VAS A VER
  Las tres estrategias aplicadas al MISMO documento, una al lado de la otra, y
  despues la prueba que importa: cual encuentra mejor la respuesta correcta.

COMO CORRERLO
    python 03_estrategias_de_chunking.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.chunking import (chunking_respetando_limites, chunking_tamano_fijo,
                             chunking_ventana_deslizante)
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.texto import contiene, normalizar
from common.ui import mostrar_configuracion, subtitulo, titulo
from common.vectores import a_matriz, buscar_mas_similares


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO = os.path.join(CARPETA, "datos", "base_conocimiento.md")



def mostrar_chunks(chunks, cuantos=3):
    """Imprime los primeros chunks recortados, para poder compararlos."""
    print("  Genero " + str(len(chunks)) + " chunks.")
    print()

    for indice in range(min(cuantos, len(chunks))):
        texto = chunks[indice].replace("\n", " ")
        palabras = len(chunks[indice].split())

        print("  --- chunk " + str(indice) + " (" + str(palabras) + " palabras) ---")
        print("  " + texto[:150] + ("..." if len(texto) > 150 else ""))
        print()


def main():
    parser = crear_parser("Las tres estrategias de chunking")
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)

    with open(ARCHIVO, "r", encoding="utf-8") as archivo:
        documento = archivo.read()

    print("[documento] " + str(len(documento.split())) + " palabras, " +
          str(len(documento)) + " caracteres")

    # =======================================================================
    titulo("ESTRATEGIA 1: TAMANO FIJO")
    # =======================================================================
    print("  Corta cada N palabras, sin mirar el contenido.")
    print()

    fijos = chunking_tamano_fijo(documento, palabras_por_chunk=80)
    mostrar_chunks(fijos)

    # Mostramos el problema concreto: el final y el principio de dos chunks
    # consecutivos, para que se vea el corte a mitad de idea.
    if len(fijos) >= 2:
        subtitulo("El problema, en vivo: mira el corte entre el chunk 0 y el 1")
        print("  Termina el chunk 0:")
        print("    ..." + " ".join(fijos[0].split()[-12:]))
        print()
        print("  Empieza el chunk 1:")
        print("    " + " ".join(fijos[1].split()[:12]) + "...")
        print()
        print("  Si la idea importante quedo justo ahi, se partio al medio y")
        print("  ninguno de los dos chunks la tiene completa.")

    # =======================================================================
    titulo("ESTRATEGIA 2: VENTANA DESLIZANTE")
    # =======================================================================
    print("  Igual que la anterior, pero cada chunk repite el final del previo.")
    print()

    deslizantes = chunking_ventana_deslizante(documento, palabras_por_chunk=80,
                                              solapamiento=20)
    mostrar_chunks(deslizantes)

    subtitulo("El costo del solapamiento")
    print("  tamano fijo        : " + str(len(fijos)) + " chunks")
    print("  ventana deslizante : " + str(len(deslizantes)) + " chunks")

    if len(fijos) > 0:
        aumento = round((len(deslizantes) / len(fijos) - 1) * 100)
        print()
        print("  Son " + str(aumento) + "% mas chunks para el mismo documento.")
        print("  Eso es " + str(aumento) + "% mas de embeddings que pagar, guardar")
        print("  y comparar en cada busqueda. El solapamiento no es gratis.")

    # =======================================================================
    titulo("ESTRATEGIA 3: RESPETANDO LIMITES")
    # =======================================================================
    print("  Junta oraciones enteras hasta acercarse al tamano objetivo.")
    print("  Nunca corta una oracion por la mitad.")
    print()

    limites = chunking_respetando_limites(documento, palabras_por_chunk=80)
    mostrar_chunks(limites)

    subtitulo("Los tamanos quedan desparejos, y esta bien")
    tamanos = []
    for chunk in limites:
        tamanos.append(len(chunk.split()))

    print("  Palabras por chunk: " + str(tamanos))
    print()
    print("  minimo: " + str(min(tamanos)) + " | maximo: " + str(max(tamanos)) +
          " | promedio: " + str(round(sum(tamanos) / len(tamanos))))
    print()
    print("  Esa desprolijidad es el precio de que cada chunk se lea bien.")

    # =======================================================================
    titulo("LA PRUEBA QUE IMPORTA: CUAL ENCUENTRA MEJOR LA RESPUESTA")
    # =======================================================================
    # Comparar estrategias mirando los chunks es subjetivo. La forma seria es
    # medir: hacer la misma pregunta a las tres y ver cual trae la respuesta
    # correcta mas arriba.

    consulta = "cuantos dias tengo para devolver un producto"

    # La respuesta correcta esta en el manual: 30 dias corridos. Buscamos esa
    # frase para saber si el chunk recuperado la contiene de verdad.
    texto_correcto = "30 dias corridos"

    print("  Consulta: \"" + consulta + "\"")
    print("  La respuesta correcta en el manual dice: \"" + texto_correcto + "\"")
    print()

    vector_consulta = embeber(cliente, [consulta], usar_cache=not args.sin_cache)[0]

    estrategias = [
        ["tamano fijo", fijos],
        ["ventana deslizante", deslizantes],
        ["respetando limites", limites],
    ]

    for estrategia in estrategias:
        nombre = estrategia[0]
        chunks = estrategia[1]

        subtitulo(nombre + "  (" + str(len(chunks)) + " chunks)")

        vectores = embeber(cliente, chunks, usar_cache=not args.sin_cache)
        matriz = a_matriz(vectores)

        resultados = buscar_mas_similares(vector_consulta, matriz, k=3)

        posicion_del_correcto = None

        for lugar in range(len(resultados)):
            indice = resultados[lugar][0]
            puntaje = resultados[lugar][1]
            texto = chunks[indice].replace("\n", " ")

            # Marcamos con >>> el chunk que SI contiene la respuesta.
            if contiene(texto, texto_correcto):
                marca = ">>>"
                if posicion_del_correcto is None:
                    posicion_del_correcto = lugar + 1
            else:
                marca = "   "

            print("  " + marca + " #" + str(lugar + 1) + "  " +
                  str(round(puntaje, 4)) + "  " + texto[:95] + "...")

        print()

        if posicion_del_correcto is None:
            print("      La respuesta correcta NO aparecio en el top 3.")
        else:
            print("      La respuesta correcta salio en el puesto " +
                  str(posicion_del_correcto) + ".")

    # =======================================================================
    titulo("COMO ELEGIR")
    # =======================================================================
    print("""
  Estrategia            Ventaja                  Desventaja
  --------------------  -----------------------  ------------------------------
  Tamano fijo           Simple y predecible      Corta ideas al medio
  Ventana deslizante    No pierde los bordes     Mas chunks: mas caro
  Respetando limites    Cada chunk se lee solo   Tamanos desparejos

  POR DONDE EMPEZAR
    Arranca con "respetando limites". Es el mejor punto de partida para
    cualquier documento escrito por una persona: manuales, politicas, FAQs.

    Si al medir ves que se pierde informacion en los cortes, sumale
    solapamiento. Combinar las dos estrategias es lo mas comun en produccion.

  EL TAMANO DEL CHUNK ES UNA PERILLA
    Chunks grandes: mas contexto, pero el tema se diluye y la busqueda pierde
    precision.
    Chunks chicos: encuentran el parrafo exacto, pero pueden quedar sin el
    contexto necesario para entenderse.

    La referencia habitual es entre 200 y 500 tokens. No lo copies: probalo
    con tus documentos y tus preguntas, como hicimos recien.
""")


if __name__ == "__main__":
    main()
