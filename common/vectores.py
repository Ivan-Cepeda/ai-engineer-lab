"""
Las operaciones matematicas sobre vectores, escritas para entenderlas.

Todo lo que hay aca existe ya hecho en NumPy o en sklearn, en una sola linea.
Lo escribimos igual porque la idea del modulo es que entiendas QUE esta pasando
cuando una base de datos vectorial "busca". Despues de este archivo, FAISS deja
de ser magia.

Cada funcion tiene dos versiones cuando vale la pena: una escrita paso a paso
con un bucle, para leerla, y una con NumPy, que es la que se usa de verdad
porque es cientos de veces mas rapida.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Lo basico: longitud de un vector y normalizacion
# ---------------------------------------------------------------------------

def norma(vector):
    """
    La "longitud" del vector, tambien llamada norma o magnitud.

    Es el teorema de Pitagoras extendido a muchas dimensiones: la raiz
    cuadrada de la suma de todos los numeros al cuadrado.

    En 2 dimensiones, el vector (3, 4) tiene norma 5, porque 3*3 + 4*4 = 25.
    """
    suma_de_cuadrados = 0

    for numero in vector:
        suma_de_cuadrados = suma_de_cuadrados + numero * numero

    return math.sqrt(suma_de_cuadrados)


def normalizar(vector):
    """
    Devuelve el mismo vector pero con longitud 1.

    Se consigue dividiendo cada numero por la longitud total. El vector sigue
    apuntando a la misma direccion, solo cambia su tamano.

    POR QUE IMPORTA: si todos los vectores tienen longitud 1, la similitud
    coseno y el producto punto dan exactamente el mismo resultado. Eso es lo
    que hace que muchas bases vectoriales usen producto punto, que es mas
    rapido de calcular.
    """
    longitud = norma(vector)

    if longitud == 0:
        return list(vector)

    return [numero / longitud for numero in vector]


# ---------------------------------------------------------------------------
# Las dos medidas de similitud de las que habla la leccion
# ---------------------------------------------------------------------------

def producto_punto(a, b):
    """
    Multiplica los vectores posicion por posicion y suma todo.

    Es la operacion mas simple de las dos, y la mas rapida.

    OJO: depende del TAMANO de los vectores, no solo de su direccion. Un
    vector que apunta al mismo lado pero es el doble de largo da el doble de
    producto punto. Por eso solo sirve cuando el tamano significa algo.
    """
    if len(a) != len(b):
        raise ValueError(
            "Los vectores tienen que tener la misma cantidad de dimensiones. "
            "Recibi " + str(len(a)) + " y " + str(len(b)) + ". "
            "La causa mas comun es mezclar embeddings de dos modelos distintos."
        )

    total = 0

    for posicion in range(len(a)):
        total = total + a[posicion] * b[posicion]

    return total


def similitud_coseno(a, b):
    """
    Mide el angulo entre dos vectores, ignorando su tamano.

    Es el producto punto dividido por las dos longitudes. Al dividir, el tamano
    se cancela y queda solo la direccion.

    El resultado va de -1 a 1:
        1  = apuntan exactamente al mismo lado (mismo significado)
        0  = perpendiculares (no tienen nada que ver)
       -1  = apuntan al lado opuesto (raro en la practica)

    Es la opcion por defecto para comparar embeddings.
    """
    longitud_a = norma(a)
    longitud_b = norma(b)

    if longitud_a == 0 or longitud_b == 0:
        return 0.0

    return producto_punto(a, b) / (longitud_a * longitud_b)


def distancia_euclidiana(a, b):
    """
    La distancia "en linea recta" entre los dos puntos.

    A diferencia de las dos anteriores, aca MAS CHICO es MEJOR: cero significa
    que son identicos. Es una distancia, no una similitud.

    Se incluye porque algunas bases vectoriales la usan por defecto y conviene
    saber que el orden de los resultados se lee al reves.
    """
    suma = 0

    for posicion in range(len(a)):
        diferencia = a[posicion] - b[posicion]
        suma = suma + diferencia * diferencia

    return math.sqrt(suma)


# ---------------------------------------------------------------------------
# Las mismas operaciones con NumPy: lo que se usa de verdad
# ---------------------------------------------------------------------------

def a_matriz(vectores):
    """
    Convierte una lista de vectores en una matriz de NumPy.

    Una matriz es una tabla de numeros: cada FILA es un vector. Tener todo
    junto en una matriz permite comparar contra miles de vectores de una sola
    operacion, en vez de con un bucle.
    """
    return np.array(vectores, dtype=np.float32)


def normalizar_matriz(matriz):
    """
    Normaliza todas las filas de la matriz de una sola vez.

    La linea de NumPy hace exactamente lo mismo que la funcion normalizar() de
    arriba, pero para todas las filas juntas y muchisimo mas rapido.
    """
    # axis=1 significa "calcula una norma por fila".
    # keepdims=True mantiene la forma para poder dividir sin errores.
    longitudes = np.linalg.norm(matriz, axis=1, keepdims=True)

    # Evitamos dividir por cero si algun vector fuera todo ceros.
    longitudes[longitudes == 0] = 1

    return matriz / longitudes


def buscar_mas_similares(vector_consulta, matriz, k=5, metrica="coseno"):
    """
    Encuentra los k vectores de la matriz mas parecidos al de la consulta.

    ESTO ES, EN ESENCIA, LO QUE HACE UNA BASE DE DATOS VECTORIAL.
    Se llama busqueda k-NN: k Nearest Neighbors, los k vecinos mas cercanos.

    Devuelve una lista de pares (posicion, puntaje), ordenada de mejor a peor.

    Nota sobre velocidad: esto compara contra TODOS los vectores, uno por uno.
    Se llama busqueda exhaustiva o "fuerza bruta", y es exacta. Con millones de
    vectores se vuelve lenta, y ahi entran los indices aproximados tipo FAISS,
    que sacrifican un poco de exactitud por mucha velocidad. Lo vemos en L2.
    """
    consulta = np.array(vector_consulta, dtype=np.float32)

    if consulta.shape[0] != matriz.shape[1]:
        raise ValueError(
            "La consulta tiene " + str(consulta.shape[0]) + " dimensiones y la "
            "matriz tiene " + str(matriz.shape[1]) + ". No se pueden comparar. "
            "Casi siempre esto significa que se mezclaron embeddings de dos "
            "modelos distintos."
        )

    if metrica == "coseno":
        # Normalizamos las dos partes y multiplicamos: eso ES el coseno.
        matriz_normalizada = normalizar_matriz(matriz)
        consulta_normalizada = consulta / (np.linalg.norm(consulta) or 1)

        # El @ multiplica la matriz por el vector: calcula de una sola vez el
        # producto punto de la consulta contra CADA fila.
        puntajes = matriz_normalizada @ consulta_normalizada
        mayor_es_mejor = True

    elif metrica == "producto_punto":
        puntajes = matriz @ consulta
        mayor_es_mejor = True

    elif metrica == "euclidiana":
        # Distancia: aca MENOR es mejor.
        diferencias = matriz - consulta
        puntajes = np.linalg.norm(diferencias, axis=1)
        mayor_es_mejor = False

    else:
        raise ValueError("Metrica desconocida: " + str(metrica))

    # argsort devuelve las POSICIONES que ordenarian el array de menor a mayor.
    orden = np.argsort(puntajes)

    if mayor_es_mejor:
        # Lo damos vuelta para que quede de mayor a menor.
        orden = orden[::-1]

    resultados = []

    for posicion in orden[:k]:
        resultados.append((int(posicion), float(puntajes[posicion])))

    return resultados
