# Diario de iteraciones · REINFORCE en el Snake

Nombre:
Uso de asistentes de IA (para qué y en qué partes):

## Reglas

- Cada iteración cambia **una sola cosa** respecto a otra anterior: la observación, la recompensa, un hiperparámetro, el baseline…
- La ficha se rellena **antes de lanzar** hasta la predicción incluida. El resultado y la decisión se escriben después. Haced un commit al terminar cada iteración.
- Lanzad cada iteración con `--tag itNN_nombre`, para que tenga su propia carpeta en `logs/lab_3_3`.
- **Explorar barato**: 2 semillas y 100 000 pasos. **Confirmar caro**: solo las 2–3 configuraciones finales, con 5 semillas y 200 000 pasos.
- Una iteración que no mejora, bien analizada, vale lo mismo que una que mejora.

Iteraciones obligatorias (en el orden que queráis, salvo la primera):

1. REINFORCE con la observación original (punto de partida).
2. Al menos una observación propia.
3. Una ablación: quitar un rasgo de vuestra observación y medir cuánto aporta.
4. Con y sin baseline (confirmado con 5 semillas).
5. Comparación con A2C y PPO con vuestro diseño final (5 semillas).

## Tabla resumen

Generadla con `python experiments_3/diario.py logs/lab_3_3` y pegadla aquí al final.

---

## Iteración 1 · `it01_original`

**Hipótesis.** ¿Qué creéis que limita al agente ahora mismo y por qué? Apoyaos en lo que habéis visto: causas de fin de partida, `--watch` o `--gif`.

**Cambio.** Una sola cosa. Comando exacto:

```
python experiments_3/lab_3_3_reinforce.py --obs original --reward sparse --seeds 0,1 --steps 100000 --tag it01_original
```

**Predicción** (antes de lanzar). ¿Cuántas manzanas esperáis, más o menos? ¿Cambiará cómo acaban las partidas (pared, cuerpo, hambre)? ¿Por qué?

**Resultado.** La fila de `diario.py` y lo que se ve al mirar jugar al agente.

**Decisión.** ¿Se queda o se descarta? ¿Qué probáis después y por qué?

---

## Iteración 2 · `it02_…`

**Hipótesis.**

**Cambio.**

```
python experiments_3/lab_3_3_reinforce.py … --tag it02_…
```

**Predicción.**

**Resultado.**

**Decisión.**

---

(Copiad la ficha para cada iteración nueva.)
