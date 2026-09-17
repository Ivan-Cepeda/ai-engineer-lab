"""
L2 - Ejercicio 01: la busqueda k-NN, escrita a mano.

QUE ES k-NN
  k Nearest Neighbors: los k vecinos mas cercanos. Dado un vector de consulta,
  encontrar los k vectores mas parecidos de tu coleccion.

  Eso es, literalmente, todo lo que hace una base de datos vectorial. Despues
  le agregan velocidad, persistencia y filtros, pero el corazon es esto.

LO QUE VAS A VER
  Primero lo escribimos con un bucle de Python, linea por linea, para que no
  quede nada magico. Despues lo hacemos con NumPy y medimos la diferencia de
  velocidad. Y al final vemos por que, con millones de vectores, ni siquiera
  NumPy alcanza y aparecen los indices aproximados.

COMO CORRERLO
    python 01_busqueda_knn_desde_cero.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.chunking import chunking_respetando_limites, crear_chunks_con_metadata
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.ui import mostrar_configuracion, subtitulo, titulo
from common.vectores import a_matriz, buscar_mas_similares, similitud_coseno


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO = os.path.join(CARPETA, "datos", "base_conocimiento.md")


def buscar_con_bucle(vector_consulta, todos_los_vectores, k=3):
    """
    La busqueda k-NN escrita de la forma mas explicita posible.

    PASO A PASO:
      1. Comparar la consulta contra CADA vector guardado, uno por uno.
      2. Guardar cada puntaje junto con la posicion a la que corresponde.
      3. Ordenar de mayor a menor.
      4. Devolver los primeros k.

    A esto se le llama busqueda EXHAUSTIVA o "por fuerza bruta". Es exacta:
    encuentra siempre el mejor resultado, porque los miro todos.
    """
    puntajes = []

    # Paso 1 y 2: comparar contra cada uno.
    for posicion in range(len(todos_los_vectores)):
        puntaje = similitud_coseno(vector_consulta, todos_los_vectores[posicion])
        puntajes.append([posicion, puntaje])

    # Paso 3: ordenar. La funcion de abajo le dice a sort() que ordene
    # mirando el segundo elemento de cada par (el puntaje).
    # reverse=True lo pone de mayor a menor.
    puntajes.sort(key=lambda par: par[1], reverse=True)

    # Paso 4: los primeros k.
    return puntajes[:k]


def main():
    parser = crear_parser("Busqueda k-NN escrita a mano")
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

    consulta = "que garantia tienen los notebooks"
    vector_consulta = embeber(cliente, [consulta], usar_cache=not args.sin_cache)[0]

    # =======================================================================
    titulo("1) LA BUSQUEDA, CON UN BUCLE COMUN")
    # =======================================================================

    print("  Consulta: \"" + consulta + "\"")
    print("  Coleccion: " + str(len(vectores)) + " chunks")
    print()

    resultados = buscar_con_bucle(vector_consulta, vectores, k=3)

    for lugar in range(len(resultados)):
        posicion = resultados[lugar][0]
        puntaje = resultados[lugar][1]
        texto = chunks[posicion]["texto"].replace("\n", " ")

        print("  #" + str(lugar + 1) + "  " + str(round(puntaje, 4)) + "  " +
              chunks[posicion]["chunk_id"])
        print("      " + texto[:100] + "...")
        print()

    print("  Eso es una base de datos vectorial. Sin bibliotecas, sin servidor:")
    print("  comparar contra todos y quedarse con los mejores.")

    # =======================================================================
    titulo("2) LO MISMO CON NUMPY, Y POR QUE ES TAN DISTINTO")
    # =======================================================================

    matriz = a_matriz(vectores)

    print("  La matriz tiene forma " + str(matriz.shape) + ":")
    print("    " + str(matriz.shape[0]) + " filas (una por chunk)")
    print("    " + str(matriz.shape[1]) + " columnas (las dimensiones del embedding)")
    print()
    print("  Con el bucle hacemos " + str(matriz.shape[0]) + " comparaciones de " +
          str(matriz.shape[1]) + " numeros cada una, una despues de la otra.")
    print()
    print("  NumPy hace las " + str(matriz.shape[0] * matriz.shape[1]) +
          " multiplicaciones de una sola vez, en codigo compilado y usando")
    print("  las instrucciones del procesador que operan sobre varios numeros")
    print("  en paralelo. A eso se le llama vectorizar.")

    resultados_numpy = buscar_mas_similares(vector_consulta, matriz, k=3)

    subtitulo("Los dos metodos dan lo mismo?")

    iguales = True

    for lugar in range(3):
        posicion_bucle = resultados[lugar][0]
        posicion_numpy = resultados_numpy[lugar][0]

        print("  #" + str(lugar + 1) + "  bucle -> chunk " + str(posicion_bucle) +
              "   numpy -> chunk " + str(posicion_numpy))

        if posicion_bucle != posicion_numpy:
            iguales = False

    print()
    if iguales:
        print("  Identicos. NumPy no cambia el resultado, solo la velocidad.")

    # =======================================================================
    titulo("3) CUANTO MAS RAPIDO ES, MEDIDO")
    # =======================================================================
    # Con 13 chunks la diferencia no se nota. Generamos vectores falsos para
    # simular una coleccion grande y que se vea el problema real.

    cantidad_falsa = 20000
    dimensiones = len(vectores[0])

    print("  Generamos " + str(cantidad_falsa) + " vectores al azar de " +
          str(dimensiones) + " dimensiones,")
    print("  para simular una base de conocimiento grande.")
    print()

    # np.random.rand crea numeros al azar entre 0 y 1.
    matriz_grande = np.random.rand(cantidad_falsa, dimensiones).astype(np.float32)
    consulta_falsa = np.random.rand(dimensiones).astype(np.float32)

    # --- con bucle ---
    # Solo medimos sobre una parte, porque con 20000 el bucle tarda demasiado.
    muestra = 1000
    lista_muestra = matriz_grande[:muestra].tolist()

    inicio = time.time()
    buscar_con_bucle(consulta_falsa.tolist(), lista_muestra, k=3)
    tiempo_bucle_muestra = time.time() - inicio

    # Estimamos cuanto tardaria con todos.
    tiempo_bucle_estimado = tiempo_bucle_muestra * (cantidad_falsa / muestra)

    # --- con numpy ---
    inicio = time.time()
    buscar_mas_similares(consulta_falsa, matriz_grande, k=3)
    tiempo_numpy = time.time() - inicio

    print("  Con bucle de Python (" + str(muestra) + " vectores): " +
          str(round(tiempo_bucle_muestra, 3)) + " s")
    print("  Estimado para " + str(cantidad_falsa) + " vectores:   " +
          str(round(tiempo_bucle_estimado, 2)) + " s")
    print()
    print("  Con NumPy (" + str(cantidad_falsa) + " vectores):        " +
          str(round(tiempo_numpy, 4)) + " s")
    print()

    if tiempo_numpy > 0:
        veces = round(tiempo_bucle_estimado / tiempo_numpy)
        print("  NumPy es aproximadamente " + str(veces) + " veces mas rapido.")

    print()
    print("  REGLA: en cuanto trabajes con vectores, no uses bucles de Python.")
    print("  No es una cuestion de estilo: es la diferencia entre una busqueda")
    print("  instantanea y una que el usuario abandona.")

    # =======================================================================
    titulo("4) DONDE NUMPY TAMBIEN SE QUEDA CORTO")
    # =======================================================================

    por_millon = tiempo_numpy * (1000000 / cantidad_falsa)

    print("  Extrapolando lo que acabamos de medir:")
    print()
    print("    1 millon de vectores  ->  " + str(round(por_millon, 3)) + " s por busqueda")
    print("    10 millones           ->  " + str(round(por_millon * 10, 2)) + " s por busqueda")
    print()
    print("  Y eso es por CADA consulta de CADA usuario. Con 100 usuarios")
    print("  simultaneos, no alcanza.")
    print()
    print("  El problema de fondo: la busqueda exhaustiva compara contra TODOS.")
    print("  Su costo crece en linea recta con la cantidad de vectores.")
    print()
    print("  LA SOLUCION: indices aproximados (ANN, Approximate Nearest")
    print("  Neighbors). En vez de mirar todos, agrupan los vectores de antemano")
    print("  y en la busqueda miran solo los grupos prometedores.")
    print()
    print("  El precio es que son APROXIMADOS: de vez en cuando se pierden el")
    print("  mejor resultado. A cambio son cientos de veces mas rapidos.")
    print()
    print("  FAISS, pgvector, Pinecone y Qdrant hacen exactamente eso. Lo vemos")
    print("  en los ejercicios 03 y 04.")

    # =======================================================================
    titulo("PARA LLEVARSE")
    # =======================================================================
    print("""
  * Una base de datos vectorial, en el fondo, compara tu consulta contra los
    vectores guardados y devuelve los mas parecidos. Nada mas que eso.

  * La busqueda exhaustiva es EXACTA pero su costo crece con la coleccion.

  * NumPy la hace cientos de veces mas rapida sin cambiar el resultado,
    porque opera sobre toda la matriz de una vez en lugar de iterar.

  * A partir de cierto tamano (cientos de miles de vectores) hay que pasar a
    indices aproximados, que cambian un poco de exactitud por mucha velocidad.

  * Hasta unos 50.000 vectores, NumPy sobra. No montes infraestructura que
    todavia no necesitas.
""")


if __name__ == "__main__":
    main()
