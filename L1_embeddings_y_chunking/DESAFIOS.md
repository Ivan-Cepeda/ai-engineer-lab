# L1 — Desafíos

Todos deben funcionar con ambos proveedores.

---

## Desafío 1 — El detector de duplicados
**Dificultad: baja** · *Ejercicios 01 y 02*

Escribí un programa que reciba una lista de tickets de soporte y encuentre los
que dicen lo mismo con otras palabras.

- Escribí 12 tickets, incluyendo 3 pares que sean el mismo problema redactado
  distinto.
- Calculá la similitud de todos contra todos.
- Mostrá los pares por encima de un umbral.

**La parte difícil:** ¿qué umbral usás? Probá varios y contá cuántos pares
correctos y cuántos falsos detecta cada uno. Vas a descubrir que no hay un
número mágico, y que el umbral depende de tu modelo y de tus datos.

---

## Desafío 2 — Clasificar sin entrenar nada
**Dificultad: media** · *Ejercicio 01*

Clasificá tickets en categorías usando **sólo** embeddings, sin llamar a un
modelo de chat.

1. Escribí una descripción de cada categoría (`facturacion`, `tecnico`,
   `cuenta`, `envios`).
2. Embebé las descripciones.
3. Para cada ticket, calculá contra cuál descripción se parece más.

Medí la exactitud sobre 20 tickets etiquetados a mano.

**Pregunta:** compará el costo y la latencia contra hacerlo con un modelo de
chat (como en M1-L3). ¿En qué casos conviene cada uno?

---

## Desafío 3 — El buscador de tu propia documentación
**Dificultad: media** · *Ejercicios 03 y 04*

Agarrá un documento **tuyo** — apuntes, un manual, la documentación de un
proyecto — y armá un buscador semántico.

- Probá las tres estrategias de chunking sobre él.
- Escribí 10 preguntas cuya respuesta sepas dónde está.
- Medí en qué puesto aparece el fragmento correcto con cada estrategia.

Entregá una tabla: estrategia · cuántas veces el correcto salió #1 · cuántas
veces entró al top-3 · cantidad de chunks generados.

---

## Desafío 4 — Calibrar el tamaño del chunk
**Dificultad: media-alta** · *Ejercicio 03*

El tamaño del chunk es una perilla y nadie te puede decir el valor correcto.
Encontralo midiendo.

- Probá 40, 80, 150, 300 y 500 palabras por chunk.
- Con el mismo set de preguntas del desafío 3, medí exactitud en top-3.
- Anotá también cuántos chunks genera cada tamaño (eso es costo).

Graficá o tabulá el resultado. **Vas a encontrar un punto óptimo**, y no va a
estar en ninguno de los extremos. Explicá por qué cae donde cae.

---

## Desafío 5 — Arreglar el chunk ruidoso
**Dificultad: alta** · *Ejercicio 04*

En el ejercicio 04 detectamos chunks genéricos que se cuelan en todas las
búsquedas. Ahora arreglalo, probando tres soluciones:

1. Sacar del índice los chunks de menos de N palabras.
2. Pegarle a cada chunk el título de su sección antes de embeberlo, y ver si
   mejora la precisión.
3. Embeber una cosa y mostrar otra: indexar "título + texto" pero devolverle al
   usuario sólo el texto.

Medí las tres contra tu set de preguntas y quedate con la que gane.

**Pregunta:** la opción 3 rompe la intuición de que "lo que buscás es lo que
mostrás". ¿Por qué funciona igual?
