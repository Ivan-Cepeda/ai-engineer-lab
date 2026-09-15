"""
Normalizacion de texto para comparar.

POR QUE EXISTE ESTE ARCHIVO
  Empezo como una funcion de seis lineas copiada en cuatro ejercicios. Cuando
  apareció un bug (ver abajo), hubo que arreglarlo en los cuatro lugares. Eso
  es exactamente la deuda tecnica que genera la duplicacion, asi que la funcion
  se mudo aca y ahora todos la importan.

EL BUG QUE LA HIZO NECESARIA
  Los ejercicios comprueban si un fragmento recuperado contiene la respuesta
  correcta, buscando un texto dentro de otro. Eso falla por dos motivos que no
  tienen nada que ver con la busqueda:

  1. LAS TILDES. El manual dice "garantia" con tilde; la cadena que buscamos,
     sin. Para Python son dos textos distintos.

  2. LOS SALTOS DE LINEA. El manual esta escrito en Markdown con las lineas
     cortadas a 80 caracteres. La frase "cuesta 20 dolares adicionales" en el
     archivo es en realidad "cuesta 20\\ndolares adicionales". Buscarla con un
     espacio no encuentra nada.

  El segundo es especialmente traicionero: el codigo reportaba "la respuesta
  correcta no aparece" cuando en realidad la habia encontrado perfecto. Estaba
  midiendo mal, no buscando mal.
"""


def normalizar(texto):
    """
    Deja el texto listo para comparar: minusculas, sin tildes y con los
    espacios colapsados en uno solo.

    Se usa SOLO para verificar resultados en los ejercicios, nunca sobre el
    texto que se le muestra al usuario ni sobre el que se embebe.
    """
    tildes = [["a", "á"], ["e", "é"], ["i", "í"],
              ["o", "ó"], ["u", "ú"], ["n", "ñ"], ["u", "ü"]]

    texto = texto.lower()

    for par in tildes:
        texto = texto.replace(par[1], par[0])

    # split() sin argumentos parte por CUALQUIER espacio en blanco (espacios,
    # tabulaciones, saltos de linea) y descarta los vacios. Volver a unir con
    # " " deja todo con un solo espacio entre palabras.
    return " ".join(texto.split())


def contiene(texto_grande, texto_buscado):
    """
    Dice si un texto contiene al otro, ignorando tildes y saltos de linea.

    Es la funcion que usan los ejercicios para saber si el fragmento
    recuperado tiene de verdad la respuesta correcta.
    """
    return normalizar(texto_buscado) in normalizar(texto_grande)
