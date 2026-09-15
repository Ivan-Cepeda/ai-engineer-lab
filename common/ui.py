"""
Funciones para que la salida en la terminal se lea ordenada.

No tienen nada que ver con embeddings ni con IA: son ayudas de impresion para
no repetir el mismo codigo en cada ejercicio.
"""


def titulo(texto):
    """Titulo grande, con lineas de = arriba y abajo."""
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def subtitulo(texto):
    """Titulo mas chico, con guiones."""
    print()
    print("-" * 78)
    print(texto)
    print("-" * 78)


def mostrar_configuracion(cliente):
    """Avisa con que proveedor y modelo de embeddings se esta trabajando."""
    ultimos_4 = cliente["clave"][-4:]

    print("[config] proveedor: " + cliente["proveedor"] +
          " | embeddings: " + cliente["modelo_embedding"] +
          " | chat: " + cliente["modelo_chat"] +
          " | clave: ..." + ultimos_4)


def mostrar_uso_del_cache(cliente):
    """Muestra cuantos textos se pidieron y cuantos salieron del cache."""
    print("[cache] " + str(cliente["textos_desde_cache"]) + " textos reutilizados | " +
          str(cliente["textos_embebidos"]) + " pedidos al proveedor")


def barra(valor, maximo=1.0, ancho=30):
    """
    Dibuja una barra de texto para ver un numero de un vistazo.

    Sirve para comparar similitudes sin tener que leer decimales.
    Ejemplo: 0.69 con maximo 1.0 dibuja una barra llena al 69%.
    """
    if maximo == 0:
        proporcion = 0
    else:
        proporcion = valor / maximo

    if proporcion < 0:
        proporcion = 0
    if proporcion > 1:
        proporcion = 1

    llenos = int(proporcion * ancho)

    return "#" * llenos + "." * (ancho - llenos)
