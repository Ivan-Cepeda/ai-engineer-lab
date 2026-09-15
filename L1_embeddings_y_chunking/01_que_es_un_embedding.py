"""
L1 - Ejercicio 01: que es un embedding.

LA IDEA EN UNA FRASE
  Un embedding es una lista larga de numeros que representa el SIGNIFICADO de
  un texto. Textos que quieren decir cosas parecidas dan numeros parecidos.

POR QUE ESTO CAMBIA TODO
  Hasta ahora, para que una computadora encontrara un texto, tenias que buscar
  palabras exactas. Si el manual decia "devolucion" y el cliente escribia
  "reintegro", no habia coincidencia.

  Con embeddings, la computadora puede comparar SIGNIFICADOS. "Devolucion" y
  "reintegro" quedan cerca aunque no compartan ni una letra.

COMO CORRERLO
    python 01_que_es_un_embedding.py
    python 01_que_es_un_embedding.py --provider openai
"""

import os
import sys

# Agrega la carpeta de arriba al path para poder importar common/.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.ui import barra, mostrar_configuracion, subtitulo, titulo
from common.vectores import norma, similitud_coseno


def main():
    parser = crear_parser("Que es un embedding")
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)

    # =======================================================================
    titulo("1) COMO SE VE UN EMBEDDING POR DENTRO")
    # =======================================================================

    frase = "El dispositivo no se conecta al WiFi"
    vector = embeber(cliente, [frase], usar_cache=not args.sin_cache)[0]

    print("  Texto: " + frase)
    print()
    print("  Se convirtio en una lista de " + str(len(vector)) + " numeros.")
    print()
    print("  Los primeros 8:")

    for posicion in range(8):
        print("    posicion " + str(posicion) + ": " + str(round(vector[posicion], 6)))

    print("    ... y " + str(len(vector) - 8) + " numeros mas")
    print()
    print("  Ningun numero por si solo significa nada. No hay una posicion que")
    print("  sea 'lo tecnico' u 'otra' que sea 'lo enojado'. El significado esta")
    print("  repartido entre los " + str(len(vector)) + " numeros a la vez.")

    # =======================================================================
    titulo("2) LA LONGITUD DEL VECTOR")
    # =======================================================================

    longitud = norma(vector)
    print("  Longitud (norma) del vector: " + str(round(longitud, 6)))
    print()

    # Muchos proveedores devuelven los vectores ya normalizados, o sea con
    # longitud exactamente 1. Eso tiene una consecuencia practica importante,
    # que vemos en el ejercicio 02.
    if abs(longitud - 1.0) < 0.01:
        print("  Es (practicamente) 1. Tu proveedor devuelve los vectores YA")
        print("  NORMALIZADOS: todos tienen la misma longitud y lo unico que")
        print("  cambia entre ellos es hacia donde apuntan.")
        print()
        print("  Eso importa, y en el ejercicio 02 vas a ver por que.")
    else:
        print("  No es 1: los vectores de este modelo tienen longitudes")
        print("  distintas entre si. Guardate el dato para el ejercicio 02.")

    # =======================================================================
    titulo("3) LA PRUEBA DE QUE CAPTURA SIGNIFICADO")
    # =======================================================================

    # Elegimos las frases a proposito para que se vea el fenomeno:
    # la 1 y la 2 no comparten casi ninguna palabra pero dicen lo mismo.
    frases = [
        "El dispositivo no se conecta al WiFi",      # 0
        "Mi router no me da internet",               # 1  mismo tema, otras palabras
        "No puedo conectarme a la red inalambrica",  # 2  mismo tema otra vez
        "Quiero devolver un producto fallado",       # 3  otro tema de la misma tienda
        "Como hago milanesas napolitanas",           # 4  nada que ver
    ]

    vectores = embeber(cliente, frases, usar_cache=not args.sin_cache, mostrar=True)

    subtitulo("Que tan parecida es cada frase a la primera")
    print("  Referencia: \"" + frases[0] + "\"")
    print()

    for posicion in range(1, len(frases)):
        puntaje = similitud_coseno(vectores[0], vectores[posicion])

        print("  " + barra(puntaje) + "  " + str(round(puntaje, 4)))
        print("      " + frases[posicion])
        print()

    print("  MIRA LA FRASE 1: \"Mi router no me da internet\".")
    print("  No comparte NI UNA palabra con la referencia (ni 'dispositivo',")
    print("  ni 'conecta', ni 'WiFi'), y sin embargo es de las mas parecidas.")
    print()
    print("  Una busqueda por palabras clave no la habria encontrado nunca.")

    # =======================================================================
    titulo("4) EL DETALLE QUE CONFUNDE A TODO EL MUNDO")
    # =======================================================================

    puntaje_pizza = similitud_coseno(vectores[0], vectores[4])

    print("  Fijate el puntaje de la milanesa: " + str(round(puntaje_pizza, 4)))
    print()
    print("  Uno esperaria algo cercano a 0, porque no tiene NADA que ver con")
    print("  un problema de WiFi. Y sin embargo da un numero alto.")
    print()
    print("  Esto es normal y hay que entenderlo bien:")
    print()
    print("  * Los puntajes de similitud NO son porcentajes. Un 0.57 no")
    print("    significa '57% parecido'.")
    print()
    print("  * Cada modelo tiene su propio 'piso'. En algunos modelos dos")
    print("    textos sin relacion dan 0.1; en otros dan 0.6.")
    print()
    print("  * Lo que importa es el ORDEN, no el numero. La frase del router")
    print("    puntua mas alto que la milanesa, y eso es lo que usa el sistema")
    print("    para decidir que mostrar.")
    print()
    print("  CONSECUENCIA PRACTICA: nunca copies un umbral fijo de un tutorial")
    print("  (\"filtra todo lo que este por debajo de 0.8\"). Ese numero depende")
    print("  del modelo. Tenes que medirlo con tus propios datos.")

    # =======================================================================
    titulo("PARA LLEVARSE")
    # =======================================================================
    print("""
  * Un embedding es una lista de cientos o miles de numeros que representa el
    significado de un texto.

  * Textos que significan lo mismo dan vectores parecidos, aunque no compartan
    ni una palabra. Eso es lo que hace posible la busqueda semantica.

  * Los puntajes de similitud no son porcentajes y no son comparables entre
    modelos distintos. Sirven para ORDENAR, no para medir en absoluto.

  * Dos embeddings de modelos distintos NO se pueden comparar entre si. Si
    cambias de modelo, tenes que volver a embeber todo.
""")


if __name__ == "__main__":
    main()
