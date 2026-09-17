# L2 — Desafíos

---

## Desafío 1 — Agregarle "no encontré nada" a la base
**Dificultad: baja** · *Ejercicio 02*

La búsqueda siempre devuelve k resultados. Hacé que sepa rendirse.

- Agregá un parámetro `umbral_minimo` a la función `buscar`.
- Si el mejor resultado no lo supera, devolvé una lista vacía.
- Encontrá el umbral midiendo: escribí 10 consultas con respuesta y 10 sin
  respuesta en el manual, y buscá el valor que mejor las separe.

**Lo que vas a descubrir:** que los dos grupos se solapan. No hay un umbral
perfecto. Elegí uno y justificá qué error preferís cometer.

---

## Desafío 2 — Actualización incremental
**Dificultad: media** · *Ejercicio 02*

Re-embeber todo cada vez que cambia un documento es carísimo. Hacelo bien:

- Guardá junto a cada chunk un hash de su texto.
- Al re-indexar, compará hashes y re-embebé **sólo** los chunks que cambiaron.
- Mostrá cuántos se reutilizaron y cuántos se pidieron de nuevo.

Probalo editando una sola línea del manual y volviendo a indexar.

---

## Desafío 3 — Medí el precio de lo aproximado
**Dificultad: media-alta** · *Ejercicio 01*

Instalá `faiss-cpu` e implementá la misma búsqueda con un índice aproximado
(`IndexIVFFlat`).

- Compará contra la búsqueda exhaustiva sobre 50.000 vectores.
- Medí velocidad **y** *recall*: de los 10 resultados verdaderos, ¿cuántos
  encuentra el índice aproximado?
- Probá distintos valores de `nprobe` y graficá velocidad contra recall.

**La pregunta:** ¿cuánta exactitud estás dispuesto a perder por cuánta
velocidad? No hay respuesta universal — depende de si tu sistema recomienda
películas o busca jurisprudencia.

---

## Desafío 4 — Ajustar el peso de la búsqueda híbrida
**Dificultad: alta** · *Ejercicio 03*

RRF trata las dos búsquedas como iguales. No siempre conviene.

- Agregá un peso: `puntaje = peso * (1/(k+pos_semantica)) + (1-peso) * (1/(k+pos_palabras))`.
- Armá 20 consultas: 10 conceptuales y 10 con datos exactos (códigos, precios).
- Encontrá el peso que maximiza el MRR sobre las 20 juntas.

**Extensión difícil:** ¿se puede elegir el peso por consulta en vez de fijo? Una
consulta con un código de producto debería inclinarse a palabras clave. Escribí
una regla que lo detecte y medí si mejora.
