"""
L1 - Ejercicio 04: chunks con metadata, y el ruido que arruina la busqueda.

DOS TEMAS EN UNO
  1. Por que cada chunk necesita datos de trazabilidad (metadata).
  2. Un problema real que vas a ver apenas armes tu primer buscador: chunks
     que aparecen primeros sin merecerlo.

POR QUE LA METADATA NO ES OPCIONAL
  Sin ella, cuando tu sistema responde algo no podes saber de donde lo saco.
  Con ella podes citar la fuente, auditar, actualizar solo lo que cambio y
  encontrar que chunk esta arruinando los resultados. Eso ultimo es
  exactamente lo que vamos a hacer en la segunda parte.

COMO CORRERLO
    python 04_chunks_con_metadata.py
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.chunking import chunking_respetando_limites, crear_chunks_con_metadata
from common.config import ErrorDeConfiguracion, crear_parser
from common.embeddings import crear_cliente_desde_argumentos, embeber
from common.ui import mostrar_configuracion, mostrar_uso_del_cache, subtitulo, titulo
from common.vectores import a_matriz, buscar_mas_similares


CARPETA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO = os.path.join(CARPETA, "datos", "base_conocimiento.md")


def main():
    parser = crear_parser("Chunks con metadata y ruido en la busqueda")
    args = parser.parse_args()

    try:
        cliente = crear_cliente_desde_argumentos(args)
    except ErrorDeConfiguracion as error:
        print(error)
        return

    mostrar_configuracion(cliente)

    with open(ARCHIVO, "r", encoding="utf-8") as archivo:
        documento = archivo.read()

    # =======================================================================
    titulo("1) UN CHUNK NO ES SOLO TEXTO")
    # =======================================================================

    chunks = crear_chunks_con_metadata("manual_v2.3", documento,
                                       chunking_respetando_limites)

    print("  Se generaron " + str(len(chunks)) + " chunks con metadata.")
    print()
    print("  Asi se ve uno por dentro:")
    print()

    ejemplo = dict(chunks[2])
    # Recortamos el texto solo para que entre en pantalla.
    ejemplo["texto"] = ejemplo["texto"][:70] + "..."

    print(json.dumps(ejemplo, ensure_ascii=False, indent=2))

    print()
    print("  Para que sirve cada campo:")
    print()
    print("    chunk_id         identificar este pedazo exacto en los registros")
    print("    documento_id     saber de que archivo salio")
    print("    indice           el orden: permite mostrar el chunk anterior y")
    print("                     el siguiente si hace falta mas contexto")
    print("    caracter_inicio  ubicar la cita exacta en el documento original")
    print("    palabras         controlar cuanto contexto le mandas al modelo")

    # =======================================================================
    titulo("2) LO QUE HABILITA TENER ESTOS DATOS")
    # =======================================================================
    print("""
  CITAR LA FUENTE
    Tu sistema puede decir "esto sale del manual v2.3, caracteres 1024 a 1523"
    en vez de pedirle al usuario que confie.

  AUDITAR
    Si dentro de seis meses alguien reclama "el bot me dijo otra cosa", podes
    reconstruir exactamente que texto vio el modelo ese dia.

  ACTUALIZAR DE A POCO
    Cuando cambia un documento, re-embebes solo ese documento_id. Sin metadata
    tendrias que rehacer toda la base.

  DEPURAR
    Encontrar el chunk especifico que esta trayendo malos resultados. Es lo
    que hacemos justo abajo.
""")

    # =======================================================================
    titulo("3) EL PROBLEMA QUE NADIE TE AVISA: EL CHUNK RUIDOSO")
    # =======================================================================

    textos = []
    for chunk in chunks:
        textos.append(chunk["texto"])

    vectores = embeber(cliente, textos, usar_cache=not args.sin_cache, mostrar=True)
    matriz = a_matriz(vectores)

    # Hacemos varias consultas de temas bien distintos y contamos cuantas
    # veces gana cada chunk. Un chunk que gana SIEMPRE es sospechoso.
    consultas = [
        "cuantos dias tengo para devolver un producto",
        "cuanto sale el envio al interior",
        "que garantia tiene un notebook",
        "no me llega el mail para recuperar la contrasena",
        "me rechazaron la tarjeta de credito",
        "el monitor no muestra imagen",
    ]

    vectores_consulta = embeber(cliente, consultas, usar_cache=not args.sin_cache)

    veces_en_el_top = {}

    subtitulo("El primer resultado de cada consulta")

    for posicion in range(len(consultas)):
        resultados = buscar_mas_similares(vectores_consulta[posicion], matriz, k=1)
        indice_ganador = resultados[0][0]
        puntaje = resultados[0][1]

        chunk_ganador = chunks[indice_ganador]
        texto_corto = chunk_ganador["texto"].replace("\n", " ")[:60]

        print("  \"" + consultas[posicion] + "\"")
        print("     -> " + chunk_ganador["chunk_id"] + "  (" +
              str(round(puntaje, 4)) + ")  " + texto_corto + "...")
        print()

        clave = chunk_ganador["chunk_id"]

        if clave in veces_en_el_top:
            veces_en_el_top[clave] = veces_en_el_top[clave] + 1
        else:
            veces_en_el_top[clave] = 1

    subtitulo("Cuantas veces gano cada chunk")

    hay_sospechoso = False

    for chunk_id in veces_en_el_top:
        cantidad = veces_en_el_top[chunk_id]

        marca = ""
        if cantidad >= 3:
            marca = "   <-- SOSPECHOSO"
            hay_sospechoso = True

        print("  " + chunk_id + ": " + str(cantidad) + " de " +
              str(len(consultas)) + marca)

    print()

    if hay_sospechoso:
        print("  Un chunk que gana en consultas de temas COMPLETAMENTE distintos")
        print("  no es que sea muy bueno: es que es muy generico.")
        print()
        print("  El caso tipico es el encabezado del documento, que dice cosas")
        print("  como 'Manual de operaciones - atencion al cliente'. Ese texto")
        print("  se parece un poco a TODO, y entonces se cuela en todas las")
        print("  busquedas ocupando un lugar que le corresponde a otro chunk.")
        print()
        print("  Gracias a la metadata podemos identificarlo exactamente y")
        print("  decidir que hacer: sacarlo del indice, o pegarle el encabezado")
        print("  a cada chunk como contexto en vez de dejarlo suelto.")
    else:
        print("  No hay ningun chunk dominando. En este documento la particion")
        print("  quedo pareja, que es lo que uno quiere.")

    print()
    mostrar_uso_del_cache(cliente)

    # =======================================================================
    titulo("PARA LLEVARSE")
    # =======================================================================
    print("""
  * Un chunk sin metadata es un texto huerfano: sirve para buscar, no sirve
    para responder con respaldo ni para depurar.

  * El texto que entra al indice no tiene por que ser identico al que le
    mostras al usuario. Una tecnica comun es embeber "titulo de la seccion +
    texto del chunk", para que el chunk sepa de que seccion viene.

  * Revisa que chunks ganan mas seguido. Si uno gana siempre, casi seguro es
    ruido, no calidad.

  * La metadata es barata de agregar al principio e imposible de reconstruir
    despues. Agregala desde el primer dia.
""")


if __name__ == "__main__":
    main()
