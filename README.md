# AI Engineer Lab

Ejercicios prácticos del **módulo 2** de la carrera de AI Engineering:
embeddings, bases vectoriales y RAG.

> **Publicación en curso.** Por ahora está la lección 1. Las otras tres
> (bases vectoriales, RAG y open source) se van sumando a este mismo repo.
Todo corre indistintamente con **OpenAI** o con **Google Gemini**, cambiando una
variable de entorno.

> **Se llama "lab" por una razón.** Acá nada se afirma: se mide. Cada ejercicio
> corre una prueba y te muestra el número, incluso cuando el número contradice
> lo que dicen los tutoriales. Hay al menos tres casos en este repo donde eso
> pasa, y están señalados en vez de disimulados.

### Si venís del módulo 1

El M1 —integración de LLMs, prompt engineering, seguridad y ética— está en un
repositorio aparte:
**https://github.com/Ivan-Cepeda/primeros_pasos_ai_engineer**

Este repo es autocontenido: no hace falta clonar el otro para correrlo. Comparte
la misma filosofía (`common/` desacoplado del proveedor, `.env` para las claves,
código sin clases ni sintaxis avanzada) pero tiene su propia copia, porque los
módulos de M2 hacen cosas que M1 no necesitaba.

---

## Qué construye este módulo

M1 terminó con un asistente que responde bien pero **inventa cuando no sabe**.
M2 es la solución a ese problema.

Al final vas a tener un sistema que, antes de responder, busca la información en
tus documentos y contesta sólo con lo que encontró — citando de dónde lo sacó.

```
L1  →  convertir texto en números que capturan significado   ← disponible
L2  →  guardar esos números y buscar en ellos rápido          (en camino)
L3  →  usar lo que encontraste para responder con fundamento  (en camino)
L4  →  hacerlo sin depender de una API, si te conviene        (en camino)
```

---

## Puesta en marcha

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
copy .env.example .env
```

Completá **una** clave en el `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=tu-clave-aca
```

- Gemini (capa gratuita): https://aistudio.google.com/apikey
- OpenAI: https://platform.openai.com/api-keys

Comprobá que funciona:

```bash
python L1_embeddings_y_chunking/01_que_es_un_embedding.py
```

---

## Sobre el nivel de Python

Igual que en M1: sólo funciones, listas, diccionarios y bucles `for`. Nada de
clases ni sintaxis avanzada. Los comentarios explican tanto el concepto de IA
como el Python que se está usando.

La única librería nueva es **NumPy**, y está explicada donde aparece: es lo que
hace que comparar contra miles de vectores tarde milisegundos en vez de minutos.

---

## Estructura

```
M2/
├── common/                      Lo compartido por todas las lecciones
│   ├── config.py                Credenciales y modelos (chat + embeddings)
│   ├── embeddings.py            Generar embeddings, con caché en disco
│   ├── vectores.py              Similitud, normalización, búsqueda k-NN
│   ├── chunking.py              Las cinco estrategias de fragmentación
│   ├── texto.py                 Normalización para comparar resultados
│   └── ui.py                    Impresión en consola
│
├── datos/
│   └── base_conocimiento.md     El manual que usan todos los ejercicios
│
├── L1_embeddings_y_chunking/
│   ├── 01_que_es_un_embedding.py
│   ├── 02_similitud_coseno_vs_punto.py
│   ├── 03_estrategias_de_chunking.py
│   └── 04_chunks_con_metadata.py
│
├── cache/                       Embeddings guardados (ignorado por git)
├── .env.example
└── requirements.txt
```

> La carpeta `PI/` que puedas tener localmente es un repositorio git aparte y
> está excluida de este repo a propósito.

---

## El caché, y por qué importa

`common/embeddings.py` guarda cada embedding en `cache/`. La primera vez que
corrés un ejercicio le pide los vectores al proveedor; después los lee del
disco.

No es sólo una comodidad: es **la técnica real**. En producción los documentos
se embeben una vez y se guardan; sólo se re-embebe lo que cambia.

Para medir tiempos reales, saltealo:

```bash
python L1_embeddings_y_chunking/03_estrategias_de_chunking.py --sin-cache
```

---

## Lo que mide cada lección

Todos los ejercicios están hechos para **medir**, no para afirmar. Algunos
resultados reales obtenidos con `gemini-embedding-001`:

| Dónde | Qué se midió | Resultado |
|-------|--------------|-----------|
| L1-02 | ¿Coseno y producto punto dan lo mismo? | Sí, porque los vectores vienen normalizados (norma = 1.0) |
| L1-05 | Las cinco estrategias de chunking | **Empate** entre tres, incluida "tamaño fijo" |

El segundo es el más incómodo: sobre un documento chico, cortar por tamaño
fijo funciona igual de bien que las estrategias avanzadas. El ejercicio explica
por qué, en vez de disimularlo.

---

## Qué se agregó desde un video externo

Después de armar el módulo se revisó el video *"Las técnicas de RAG avanzado que
uso en producción"* y se contrastó con el código. De sus recomendaciones, el módulo
ya cubría 4; las otras 9 se incorporaron. Cada archivo nuevo lo aclara en su
encabezado. Abajo, lo que aplica a L1:

| Recomendación del video | Estado |
|---|---|
| Chunks fijos rompen la coherencia | Ya estaba (L1-03) |
| Chunking semántico | **Agregado** (L1-05) |
| Chunking por estructura | **Agregado** (L1-05) |
| Reranking, transformación de consultas y el resto | Llegan con L3 |

Un hallazgo que contradice parcialmente al video, y que quedó documentado en la
salida del ejercicio en vez de disimulado: **"tamaño fijo" empata con las
estrategias avanzadas** en nuestro manual.

La explicación es la lección de fondo: con 13 fragmentos no hay suficiente ruido
para que la diferencia se note. Estas técnicas se inventaron para bases de miles
de documentos.

---

## Diferencias entre proveedores

| | OpenAI | Gemini |
|---|---|---|
| Endpoint `/embeddings` | ✅ | ✅ *(por la URL de compatibilidad)* |
| Vectores normalizados | ✅ | ✅ |
| Dimensiones por defecto | 1536 | 3072 |
| Parámetro `dimensions` | ✅ | ✅ |

**La regla que no se negocia:** embeddings de modelos distintos **no son
comparables**. Si cambiás de modelo, tenés que volver a embeber todo. La mini
base vectorial de L2-02 verifica las dimensiones justamente para que este error
falle ruidosamente en vez de corromper los resultados en silencio.

---

## Orden sugerido

Empezá por [`L1_embeddings_y_chunking/README.md`](L1_embeddings_y_chunking/README.md)
y hacé los ejercicios en orden numérico: cada uno asume el anterior.

Las lecciones 2, 3 y 4 se publican en este mismo repositorio a medida que estén
revisadas.

---

## Costo

Embeber es barato: indexar el manual completo cuesta menos de 0,0002 USD. Lo
caro de un sistema RAG es la llamada de generación del final, no los embeddings.

Con la capa gratuita de Gemini, correr todo el módulo cuesta cero. Y el caché
hace que la segunda corrida no consuma nada.
