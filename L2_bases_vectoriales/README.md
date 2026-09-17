# L2 — Bases de datos vectoriales: indexación y recuperación

Después de esta lección, FAISS y Pinecone dejan de ser magia.

## Ejercicios

| # | Archivo | Qué enseña |
|---|---------|------------|
| 01 | [`01_busqueda_knn_desde_cero.py`](01_busqueda_knn_desde_cero.py) | Qué hace realmente una base vectorial, escrito con un bucle; y por qué NumPy la vuelve viable |
| 02 | [`02_mini_base_vectorial.py`](02_mini_base_vectorial.py) | Una base vectorial completa: agregar, buscar, filtrar, borrar, persistir |
| 03 | [`03_busqueda_hibrida.py`](03_busqueda_hibrida.py) | Semántica + palabras clave, fusionadas con RRF; y qué le falta a un BM25 casero |
| 04 | [`04_chromadb.py`](04_chromadb.py) | La misma base, pero con ChromaDB: cada cosa que escribiste a mano tiene su equivalente |

```bash
python 01_busqueda_knn_desde_cero.py --sin-cache
```

El ejercicio 04 necesita una librería extra. Corre igual sin ella: detecta
que falta y explica los conceptos.

```bash
pip install chromadb
```

## Ideas que hay que llevarse

**Una base vectorial es una matriz y una multiplicación.** Comparar la consulta
contra todos los vectores guardados y quedarse con los mejores. Todo lo demás
—persistencia, filtros, escalado— se construye encima de eso (ejercicio 01).

**NumPy es 23× más rápido que un bucle, con el mismo resultado.** Medido en el
ejercicio. No es una cuestión de estilo: es la diferencia entre una búsqueda
instantánea y una que el usuario abandona (ejercicio 01).

**La búsqueda exhaustiva es exacta, pero su costo crece en línea recta.** A
partir de cientos de miles de vectores hace falta un índice aproximado, que
cambia un poco de exactitud por mucha velocidad. Eso es FAISS (ejercicio 01).

**Hasta 50.000 vectores, NumPy sobra.** No montes infraestructura que todavía no
necesitás (ejercicio 01).

**Verificá las dimensiones al agregar.** Mezclar embeddings de dos modelos
corrompe la base **en silencio**: no hay error, sólo resultados raros que no vas
a poder explicar. Es el peor tipo de bug (ejercicio 02).

**El filtrado por metadata no es un extra.** En producción casi ninguna búsqueda
es sobre toda la base: es sobre los documentos de este cliente, de este año, de
esta categoría (ejercicio 02).

**La búsqueda vectorial nunca dice "no encontré nada".** Siempre devuelve los k
más parecidos, tengan o no sentido, y con puntajes que parecen buenos. Decidir
si el mejor resultado alcanza es trabajo tuyo (ejercicio 02).

**La semántica es mala con datos exactos.** Para el modelo, `SKU-200` y
`SKU-300` significan casi lo mismo. Códigos, versiones, siglas y nombres propios
necesitan búsqueda por palabras clave (ejercicio 03).

**RRF fusiona rankings sin poder sumar sus puntajes.** Ignora los números y mira
sólo las posiciones. Un documento bien ubicado en las dos listas es una señal
mucho más fuerte que ser bueno en una sola (ejercicio 03).

**Una base vectorial de verdad no tiene ningún concepto nuevo.** El ejercicio 04
pone lado a lado lo que escribiste a mano y su equivalente en ChromaDB: `agregar`
es `add`, `buscar` es `query`, el filtro es `where`. Lo único que cambia es quién
escribe el código (ejercicio 04).

**Chroma devuelve DISTANCIAS, no similitudes.** Más bajo es mejor. Si las ordenás
de mayor a menor creyendo que son similitudes, te quedás con los peores
resultados y nada te avisa. Es la misma trampa de la distancia euclidiana en
L1-02 (ejercicio 04).

**El parámetro `hnsw:space` hay que ponerlo siempre.** Si no, Chroma usa
distancia euclidiana por defecto, que no es lo que querés para embeddings de
texto. Es el error silencioso más común al empezar (ejercicio 04).

**Los valores que usás para filtrar hay que normalizarlos al guardarlos.** Una
tilde de diferencia entre `garantías` y `garantias` hace que el filtro no
encuentre nada, sin ningún error. Este bug estuvo en la primera versión del
ejercicio 04 y quedó documentado ahí.

## Después de esto

Seguí con [`DESAFIOS.md`](DESAFIOS.md).

La lección **L3 (RAG)**, donde todo esto se convierte en un sistema que
responde preguntas con fundamento, se publica en este mismo repositorio.
