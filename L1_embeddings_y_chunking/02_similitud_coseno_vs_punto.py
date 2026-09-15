"""
L1 - Ejercicio 02: similitud coseno contra producto punto.

LA PREGUNTA QUE RESPONDE
  Ya tenemos vectores. Ahora, como medimos si dos vectores son parecidos?
  Hay dos formas principales y la leccion pide entender cuando usar cada una.

LA INTUICION, CON UNA ANALOGIA
  Imaginate cada vector como una FLECHA que sale del origen.

  El PRODUCTO PUNTO mira dos cosas a la vez: hacia donde apunta la flecha Y
  que tan larga es. Una flecha larga puntua mas alto aunque apunte parecido.

  La SIMILITUD COSENO mira SOLO la direccion. No le importa si la flecha mide
  2 o 200: pregunta unicamente "apuntan al mismo lado?".

COMO CORRERLO
    python 02_similitud_coseno_vs_punto.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.ui import barra, mostrar_configuracion, subtitulo, titulo
from common.vectores import (distancia_euclidiana, norma, normalizar,
                             producto_punto, similitud_coseno)


def main():
    parser = crear_parser("Similitud coseno contra producto punto")
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)

    # =======================================================================
    titulo("1) LA DIFERENCIA, CON NUMEROS CHIQUITOS")
    # =======================================================================
    # Antes de tocar embeddings de 3000 dimensiones, veamos el fenomeno con
    # vectores de 2 numeros, que se pueden dibujar en una hoja.

    # Estos dos apuntan EXACTAMENTE al mismo lado. El segundo es el primero
    # multiplicado por 3: la misma direccion, tres veces mas largo.
    corto = [1, 1]
    largo = [3, 3]

    # Este apunta a otro lado.
    otro = [1, 0]

    print("  Tenemos tres flechas en un plano de dos dimensiones:")
    print()
    print("    corto = [1, 1]   apunta en diagonal, longitud " + str(round(norma(corto), 3)))
    print("    largo = [3, 3]   MISMA diagonal, longitud " + str(round(norma(largo), 3)))
    print("    otro  = [1, 0]   apunta a la derecha, longitud " + str(round(norma(otro), 3)))
    print()

    subtitulo("Producto punto: mezcla direccion y tamano")
    print("  corto vs largo : " + str(producto_punto(corto, largo)))
    print("  corto vs corto : " + str(producto_punto(corto, corto)))
    print("  corto vs otro  : " + str(producto_punto(corto, otro)))
    print()
    print("  PROBLEMA: 'corto vs largo' da 6, mas que 'corto vs corto' que da 2.")
    print("  O sea que segun esta medida, el vector corto se parece MAS al")
    print("  largo que a si mismo. Eso no tiene sentido como medida de")
    print("  significado, y pasa solo porque el largo es mas grande.")

    subtitulo("Similitud coseno: solo la direccion")
    print("  corto vs largo : " + str(round(similitud_coseno(corto, largo), 4)))
    print("  corto vs corto : " + str(round(similitud_coseno(corto, corto), 4)))
    print("  corto vs otro  : " + str(round(similitud_coseno(corto, otro), 4)))
    print()
    print("  Ahora si: corto y largo dan 1.0 porque apuntan al mismo lado, y")
    print("  un vector consigo mismo tambien da 1.0. El tamano ya no molesta.")

    # =======================================================================
    titulo("2) POR QUE EN LA PRACTICA MUCHAS VECES DA IGUAL")
    # =======================================================================

    frases = [
        "El cliente quiere devolver un producto",
        "Como tramito un reintegro de mi compra",
        "El monitor no muestra imagen",
    ]

    vectores = embeber(cliente, frases, usar_cache=not args.sin_cache, mostrar=True)

    print()
    print("  Longitud de cada embedding que nos devolvio el proveedor:")

    todos_normalizados = True

    for posicion in range(len(frases)):
        longitud = norma(vectores[posicion])
        print("    frase " + str(posicion) + ": " + str(round(longitud, 6)))

        if abs(longitud - 1.0) > 0.01:
            todos_normalizados = False

    print()

    if todos_normalizados:
        print("  TODAS valen 1. El proveedor ya te los entrega normalizados.")
        print()
        print("  Y aca esta la consecuencia: si todos los vectores miden lo")
        print("  mismo, el tamano deja de influir, y entonces el producto punto")
        print("  y la similitud coseno dan EL MISMO NUMERO.")
    else:
        print("  No todas valen 1: este modelo devuelve vectores de distinta")
        print("  longitud, asi que las dos medidas SI van a diferir.")

    subtitulo("Comprobacion sobre frases reales")
    print("  Comparando: \"" + frases[0] + "\"")
    print("       contra: \"" + frases[1] + "\"")
    print()

    coseno = similitud_coseno(vectores[0], vectores[1])
    punto = producto_punto(vectores[0], vectores[1])

    print("    similitud coseno : " + str(round(coseno, 8)))
    print("    producto punto   : " + str(round(punto, 8)))
    print("    diferencia       : " + str(round(abs(coseno - punto), 10)))
    print()

    if abs(coseno - punto) < 0.0001:
        print("  Practicamente identicos, como esperabamos.")
        print()
        print("  POR ESO muchas bases de datos vectoriales usan producto punto")
        print("  por defecto: da el mismo resultado y es mas rapido de calcular,")
        print("  porque se saltea las dos divisiones.")

    # =======================================================================
    titulo("3) QUE PASA SI NO NORMALIZAS")
    # =======================================================================
    # Simulamos el caso problematico: agrandamos un vector a proposito para
    # ver como se rompe el ranking si usaras producto punto sin normalizar.

    consulta = vectores[0]

    # Tomamos el vector de la frase que NO tiene nada que ver (el monitor) y
    # lo multiplicamos por 3, como si viniera de un modelo que devuelve
    # vectores de longitud variable.
    vector_inflado = [numero * 3 for numero in vectores[2]]

    print("  Comparamos la consulta sobre devoluciones contra dos candidatos:")
    print()
    print("    A) \"" + frases[1] + "\"  (el correcto)")
    print("    B) \"" + frases[2] + "\"  (no tiene nada que ver, pero inflado x3)")
    print()

    punto_a = producto_punto(consulta, vectores[1])
    punto_b = producto_punto(consulta, vector_inflado)

    coseno_a = similitud_coseno(consulta, vectores[1])
    coseno_b = similitud_coseno(consulta, vector_inflado)

    print("  Con PRODUCTO PUNTO:")
    print("    A = " + str(round(punto_a, 4)))
    print("    B = " + str(round(punto_b, 4)))

    if punto_b > punto_a:
        print("    --> GANA B, que es el resultado equivocado.")
    else:
        print("    --> gana A, correcto.")

    print()
    print("  Con SIMILITUD COSENO:")
    print("    A = " + str(round(coseno_a, 4)))
    print("    B = " + str(round(coseno_b, 4)))

    if coseno_a > coseno_b:
        print("    --> gana A, correcto. El inflado no la enganio.")

    print()
    print("  Normalizar el vector inflado lo arregla:")
    print("    B normalizado = " +
          str(round(producto_punto(consulta, normalizar(vector_inflado)), 4)))

    # =======================================================================
    titulo("4) LA TERCERA MEDIDA: DISTANCIA EUCLIDIANA")
    # =======================================================================
    # Algunas bases vectoriales la usan por defecto y conviene reconocerla.

    print("  Es la distancia en linea recta entre los dos puntos.")
    print()
    print("  OJO CON ESTA: aca MAS CHICO es MEJOR. Es una distancia, no una")
    print("  similitud. Si ordenas los resultados de mayor a menor como con el")
    print("  coseno, te quedas con los PEORES resultados.")
    print()

    for posicion in range(1, len(frases)):
        distancia = distancia_euclidiana(vectores[0], vectores[posicion])
        coseno = similitud_coseno(vectores[0], vectores[posicion])

        print("  " + frases[posicion])
        print("    distancia euclidiana: " + str(round(distancia, 4)) + "  (menos es mejor)")
        print("    similitud coseno    : " + str(round(coseno, 4)) + "  (mas es mejor)")
        print()

    print("  Fijate que ordenan igual, solo que al reves. Con vectores")
    print("  normalizados las dos medidas son equivalentes: dan el mismo orden.")

    # =======================================================================
    titulo("COMO DECIDIR, EN LA PRACTICA")
    # =======================================================================
    print("""
  Medida               Le importa el tamano?   Cuando usarla
  -------------------  ----------------------  -------------------------------
  Similitud coseno     No                      Por defecto. Siempre que dudes.
  Producto punto       Si                      Si ya normalizaste, por velocidad
  Distancia euclidiana Si                      Si tu base vectorial la impone

  REGLA PRACTICA:
    1. Fijate si tu proveedor devuelve vectores normalizados (mira el punto 2
       de este ejercicio: si la longitud da 1, ya estan).
    2. Si lo estan, usa producto punto: mismo resultado, mas rapido.
    3. Si no lo estan o no estas seguro, usa coseno. Nunca te va a enganiar
       por el tamano del vector.
    4. Si usas distancia, acordate de ordenar al reves.
""")


if __name__ == "__main__":
    main()
