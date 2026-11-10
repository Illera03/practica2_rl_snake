# Diario de iteraciones · REINFORCE en el Snake

Nombres: Jorge Illera, Iker Argulo e Iñigo Lorite
Uso de asistentes de IA (para qué y en qué partes): Hemos usado Claude como asistente para implementar alguna parte del código y como asistente para redactar las fichas del diario a partir de nuestras decisiones. Las hipótesis, el plan de iteraciones, la elección de qué cambiar en cada una y la interpretación de los resultados las hemos discutido y validado nosotros; hemos revisado y ejecutado todo el código antes de usarlo.

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

**Hipótesis.** La política aleatoria come 0,02 manzanas por partida y dura 3,6 pasos; todas sus partidas acaban contra el cuerpo, porque una de cada cuatro acciones da la vuelta sobre el cuello. La observación original da coordenadas absolutas de la cabeza, la distancia a la manzana, la longitud y las últimas 30 acciones, pero no dice qué casillas vecinas son mortales. Para saber si un paso mata, la red tendría que combinar la posición con el tamaño del tablero (paredes) o reconstruir el cuerpo a partir de las acciones pasadas, y eso es muy difícil de aprender con REINFORCE, que tiene mucha varianza.

**Cambio.** Ninguno: punto de partida.

```
python experiments_3/lab_3_3_reinforce.py --obs original --reward sparse --seeds 0,1 --steps 100000 --tag it01_original
```

**Predicción.** Entre 0 y 2 manzanas por partida. Lo primero que se aprende es lo más fácil: no darse la vuelta (la dirección actual está en la última acción), así que las muertes contra el cuerpo deberían pasar a muertes contra la pared. Si aprendiera algo más, subiría el hambre.

**Resultado.**

| Iteración | Algoritmo | Obs | Recompensa | Semillas | Pasos | Manzanas | IC 95 % | IQM | Pared | Cuerpo | Hambre | Cambia respecto a la anterior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it01_original | REINFORCE | original | sparse | 2 | 100.013 | 0,16 | [0,07, 0,25] | 0,16 | 100 % | 0 % | 0 % | punto de partida |

La semilla 0 se queda en 0,05 manzanas y la 1 sube despacio hasta 0,30. Se cumple lo previsto: el agente aprende a no darse la vuelta (las muertes pasan del 100 % contra el cuerpo al 100 % contra la pared), pero no aprende dónde están las paredes y nunca sobrevive hasta el hambre. GIF: `logs/snake_it01_original.gif`.

**Decisión.** Se descarta la observación original. El cuello de botella es saber qué movimientos matan, así que en la siguiente iteración la observación será solo eso: un bit de peligro por acción.

---

## Iteración 2 · `it02_peligro`

**Hipótesis.** El agente de it01 muere siempre contra la pared porque no ve qué casillas vecinas son mortales. Si la observación dice directamente si cada acción lleva a pared o cuerpo, REINFORCE debería aprender a evitarlo enseguida.

**Cambio.** `--obs custom` con `SnakeObservation` = 4 bits de peligro, uno por acción en el orden de `MOVES` (1 si la casilla destino es pared o cuerpo). El cuerpo es `snake[:-1]`, porque la cola se mueve antes de comprobar el choque. No ve la manzana. Rasgo calculado declarado: peligro inmediato por acción (no es una búsqueda, solo describe las casillas vecinas).

```
python experiments_3/lab_3_3_reinforce.py --obs custom --reward sparse --seeds 0,1 --steps 100000 --tag it02_peligro
```

**Predicción.** Las muertes contra la pared y el cuerpo caerán mucho y la mayoría de partidas acabarán por hambre: con `sparse`, truncar vale 0 y chocar −1, así que sobrevivir compensa aunque no coma. Las manzanas subirán poco (entre 0,3 y 1,5), solo las que encuentre al deambular, porque no sabe dónde está la manzana.

**Resultado.**

| Iteración | Algoritmo | Obs | Recompensa | Semillas | Pasos | Manzanas | IC 95 % | IQM | Pared | Cuerpo | Hambre | Cambia respecto a la anterior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it02_peligro | REINFORCE | custom | sparse | 2 | 100.561 | 0,42 | [0,17, 0,68] | 0,42 | 0 % | 0 % | 100 % | obs |

Se cumple la predicción: las muertes por pared y cuerpo desaparecen (de 100 % pared en it01 a 0 %) y todas las partidas acaban por hambre. Las manzanas suben de 0,16 a 0,42, dentro del rango previsto, y la diferencia entre semillas es grande (0,20 frente a 0,65). Lo aprende muy pronto: a los 25 000 pasos ya está en su meseta y no mejora después. En el GIF la serpiente recorre el borde del tablero dando vueltas, pegada a la pared, y solo come si la manzana aparece en ese recorrido. Es justo lo que premia `sparse` sin información de la manzana: truncar (0) es mejor que chocar (−1). GIF: `logs/snake_it02_peligro.gif`.

**Decisión.** Se queda el peligro: resuelve la supervivencia inmediata. El límite ahora es que el agente no sabe dónde está la manzana, así que el siguiente cambio es añadir hacia dónde está.

---

## Iteración 3 · `it03_comida`

**Hipótesis.** En it02 el agente sobrevive pero no come porque no sabe dónde está la manzana: da vueltas por el borde hasta morir de hambre. Si la observación indica qué acciones le acercan a la manzana, podrá ir a por ella sin dejar de evitar el peligro.

**Cambio.** Se añaden a la observación 4 bits de comida, uno por acción en el orden de `MOVES`: 1 si esa acción reduce la distancia a la manzana (producto escalar entre el desplazamiento y el vector cabeza→manzana mayor que 0). Total 8 entradas. Rasgo calculado declarado: describe dónde está la manzana respecto a la cabeza; no tiene en cuenta obstáculos ni recomienda un camino.

```
python experiments_3/lab_3_3_reinforce.py --obs custom --reward sparse --seeds 0,1 --steps 100000 --tag it03_comida
```

**Predicción.** Un salto grande: entre 4 y 12 manzanas por partida. El hambre bajará mucho y la causa principal de muerte pasará a ser el cuerpo: al crecer, la serpiente se encerrará a sí misma porque solo ve las casillas vecinas. Algo de hambre puede quedar si entra en bucles cuando la manzana queda detrás de su cuerpo.

**Resultado.**

| Iteración | Algoritmo | Obs | Recompensa | Semillas | Pasos | Manzanas | IC 95 % | IQM | Pared | Cuerpo | Hambre | Cambia respecto a la anterior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it03_comida | REINFORCE | custom | sparse | 2 | 100.026 | 20,12 | [17,85, 22,40] | 20,12 | 0 % | 100 % | 0 % | código de la observación |

El salto es mucho mayor de lo previsto: de 0,42 a 20,1 manzanas por partida, por encima del rango de 4 a 12 que esperábamos. La parte cualitativa sí se cumple: el hambre desaparece (de 100 % a 0 %) y todas las partidas acaban contra el propio cuerpo. Infravaloramos lo suficientes que son peligro y comida juntos: con esos 8 bits basta una regla casi codiciosa ("ve hacia la manzana por una casilla segura"), que una red pequeña aprende enseguida. Las dos semillas aprenden a ritmos muy distintos: la 0 ya come 10,4 a los 25 000 pasos y la 1 sigue en 0,35 hasta que salta a 23,75 a los 50 000; es la varianza típica de REINFORCE. En el GIF la serpiente va directa a cada manzana y, cuando ya es larga (unas 20 casillas), se mete en un hueco cerrado por su propio cuerpo y no puede salir: solo ve las casillas vecinas, así que no sabe que el camino no tiene salida hasta que ya no queda ninguna. GIF: `logs/snake_it03_comida.gif`.

**Decisión.** Se queda la comida. El cuello de botella ya no es encontrar la manzana sino encerrarse a sí misma. La dirección actual, que era el siguiente paso del plan, no ataca ese problema (el cuello ya cuenta como peligro y las acciones son absolutas), así que cambiamos el orden: primero se añade información sobre el espacio libre más allá de la casilla vecina.

---

## Iteración 4 · `it04_espacio`

**Hipótesis.** En it03 todas las partidas acaban contra el cuerpo porque la serpiente se mete en huecos sin salida: el peligro solo mira una casilla. Si sabe cuánto espacio libre hay en cada dirección, podrá evitar los pasillos cortos y encerrarse menos.

**Cambio.** Se añaden 4 valores de espacio libre, uno por acción en el orden de `MOVES`: casillas libres en línea recta desde la cabeza hasta la primera pared o cuerpo, divididas por `grid_size − 1` para que queden en [0, 1]. Total 12 entradas. Rasgo calculado declarado: es un rayo en línea recta que describe el estado, no una búsqueda de caminos (no mira si el hueco tiene salida).

```
python experiments_3/lab_3_3_reinforce.py --obs custom --reward sparse --seeds 0,1 --steps 100000 --tag it04_espacio
```

**Predicción.** Mejora pequeña: entre 20 y 25 manzanas, posiblemente dentro del ruido con 2 semillas (el intervalo de it03 ya va de 17,9 a 22,4). Las muertes seguirán siendo casi todas contra el cuerpo, porque un rayo recto no ve huecos que se cierran por los lados, pero deberían llegar algo más tarde (serpiente más larga). Puede aparecer algo de hambre si se vuelve más prudente y da rodeos.

**Resultado.**

**Decisión.**

---

(Copiad la ficha para cada iteración nueva.)
