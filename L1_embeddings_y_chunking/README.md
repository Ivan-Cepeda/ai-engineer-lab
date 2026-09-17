# L1 — El poder de los embeddings y el text chunking

De "buscar palabras exactas" a "buscar significados".

## Ejercicios

| # | Archivo | Qué enseña |
|---|---------|------------|
| 01 | [`01_que_es_un_embedding.py`](01_que_es_un_embedding.py) | Qué es un vector de embedding, y por qué los puntajes de similitud no son porcentajes |
| 02 | [`02_similitud_coseno_vs_punto.py`](02_similitud_coseno_vs_punto.py) | Coseno, producto punto y distancia euclidiana: cuándo usar cada uno |
| 03 | [`03_estrategias_de_chunking.py`](03_estrategias_de_chunking.py) | Las tres estrategias de fragmentación, medidas sobre el mismo documento |
| 04 | [`04_chunks_con_metadata.py`](04_chunks_con_metadata.py) | Trazabilidad, y cómo detectar el chunk ruidoso que se cuela en todas las búsquedas |
| 05 | [`05_chunking_avanzado.py`](05_chunking_avanzado.py) | **Agregado desde el video**: chunking semántico y por estructura, con las cinco estrategias medidas juntas |

```bash
python 01_que_es_un_embedding.py --provider gemini
```

## Ideas que hay que llevarse

**Un embedding captura significado, no palabras.** "Mi router no da internet"
queda cerca de "el dispositivo no se conecta al WiFi" sin compartir una sola
palabra. Eso es lo que una búsqueda por palabras clave nunca va a encontrar
(ejercicio 01).

**Los puntajes de similitud no son porcentajes.** Dos textos sin relación pueden
dar 0.57. Cada modelo tiene su propio piso. Lo que importa es el **orden**, no
el número absoluto — y por eso nunca copies un umbral de un tutorial
(ejercicio 01).

**Si los vectores vienen normalizados, coseno y producto punto dan lo mismo.**
Por eso muchas bases vectoriales usan producto punto: mismo resultado, más
rápido. Comprobalo antes de asumirlo (ejercicio 02).

**La distancia euclidiana se lee al revés.** Más chico es mejor. Si ordenás de
mayor a menor como con el coseno, te quedás con los peores resultados
(ejercicio 02).

**Elegir una estrategia de chunking es elegir qué sacrificás.** Contexto,
precisión y costo compiten entre sí; no se pueden maximizar los tres
(ejercicio 03).

**Empezá por "respetando límites".** Es el mejor punto de partida para cualquier
documento escrito por una persona. Sumale solapamiento sólo si al medir ves que
se pierde información en los cortes (ejercicio 03).

**El solapamiento no es gratis.** Un 25% de solapamiento es un 25% más de
vectores que pagar, guardar y comparar en cada búsqueda (ejercicio 03).

**Un chunk sin metadata es un texto huérfano.** Sirve para buscar, no para
responder con respaldo ni para depurar. Agregala desde el primer día: es barata
al principio e imposible de reconstruir después (ejercicio 04).

**Revisá qué chunks ganan más seguido.** Si uno gana en consultas de temas
completamente distintos, no es que sea bueno: es que es genérico, y está
ocupando un lugar que le corresponde a otro (ejercicio 04).

## Nota sobre el ejercicio 05

Los ejercicios 01 a 04 cubren la lección. El **05 se agregó después**, a partir
del video *"Las técnicas de RAG avanzado que uso en producción"*, que recomienda
dos estrategias de chunking que la lección no menciona: la **semántica** (cortar
donde cambia el tema, medido con embeddings) y la **estructural** (cortar por los
títulos del documento).

Vale la pena correrlo aunque sea sólo por el resultado: sobre nuestro manual de
13 fragmentos, **tres estrategias empatan en MRR 1.0 — incluida "tamaño fijo"**,
justo la que el video señala como causa principal de que un RAG falle.

No es que el video se equivoque: es que un documento chico no tiene suficiente
ruido para que la diferencia se note. El ejercicio lo explica en su salida, y es
probablemente la lección más valiosa de L1.

## Después de esto

Seguí con [`DESAFIOS.md`](DESAFIOS.md). Son cinco consignas, y la número 4
—calibrar el tamaño del chunk midiendo— es la que más se parece al trabajo
real.

Después seguí con **[L2](../L2_bases_vectoriales/README.md)**, donde estos
vectores se guardan y se buscan en serio.
