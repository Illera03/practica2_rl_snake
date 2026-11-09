# experiments_3 · Unidad 3 · Métodos avanzados

Scripts de los laboratorios de la Unidad 3 (Reinforcement Learning, Máster en Computación y Sistemas Inteligentes, Universidad de Deusto).

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install swig                 # antes de box2d, si falla la instalación de LunarLander
pip install -r experiments_3/requirements.txt
python experiments_3/shower_continuous.py        # comprobación rápida del entorno
```

Ejecutad todo desde el directorio de desarrollo (el que contiene `experiments_3/`), igual que en la Unidad 2. Los resultados se guardan en `logs/` y las búsquedas de Optuna en `optuna/`.

Windows: si `import torch` falla con un error de DLL (WinError 1114), cread el entorno con un Python que no sea el de Anaconda, por ejemplo con `uv venv --python 3.12`. Para el Lab 3.6 (MuJoCo), WSL2 o Colab.

### Google Colab Pro+

Alternativa sin instalar nada: subid `experiments_3` a `Mi unidad/RL_U3/` y abrid `colab_unidad3.ipynb`. El cuaderno monta Drive (los resultados no se pierden al desconectar), instala las dependencias, abre TensorBoard, lanza varias semillas en paralelo y graba un vídeo del Snake.

## Estructura de resultados

```
logs/<lab>/<configuración>/seed_<k>/evaluations.npz   curva de evaluación (política determinista)
logs/<lab>/<configuración>/seed_<k>/best_model.zip    mejor modelo
logs/<lab>/<configuración>/seed_<k>/config.json       hiperparámetros usados
logs/tb/<lab>/                                         TensorBoard
```

Comparar configuraciones (media entre semillas, IC 95 % por bootstrap e IQM):

```bash
python experiments_3/plot_results.py logs/lab_3_2
tensorboard --logdir logs/tb
```

## Laboratorios

| Lab | Semana | Script | Qué hace | Presupuesto por defecto |
|---|---|---|---|---|
| 3.1 | 1 | `lab_3_1_dqn.py` | DQN en CartPole-v1 y LunarLander-v3; vídeo del mejor modelo | 50k / 150k pasos, 3 semillas |
| 3.2 | 1 | `lab_3_2_ablacion.py` | Ablación de DQN: red objetivo, buffer, exploración | 4 variantes × 3 semillas × 150k pasos |
| 3.3 | 2 | `lab_3_3_reinforce.py` | **Esqueleto** de REINFORCE en el Snake (entregable) | 200k pasos |
| 3.3 | 2 | `snake_design.py` | **Esqueleto** de observación y recompensa del Snake (entregable) | — |
| 3.3 | 2 | `snake_env.py` | Entorno `DeustoSnake-v0` | tablero 12 × 12 |
| 3.3 | 2 | `lab_3_3_referencia_sb3.py` | A2C y PPO de SB3 con el mismo diseño y presupuesto | 200k pasos, 5 semillas |
| 3.3 | 2 | `diario.py` | Tabla del diario de iteraciones a partir de los logs | — |
| 3.3 | 2 | `DIARIO.md` | **Plantilla** del diario de iteraciones (entregable) | — |
| 3.4 | 3 | `lab_3_4_ppo_a2c.py` | PPO frente a A2C en LunarLander-v3; barrido de clip_range | 500k pasos, 8 entornos |
| 3.5 | 3 | `lab_3_5_continuo.py` | DDPG, TD3, SAC (y PPO) en Pendulum-v1 y en la ducha continua | 20k pasos |
| 3.5 | 3 | `shower_continuous.py` | Entorno `ShowerContinuous-v0` (la ducha de la U1, continua) | 60 pasos por episodio |
| 3.6 | 4 | `lab_3_6_her.py` | HER en BitFlipping (sin MuJoCo) y en FetchReach-v4 (MuJoCo) | 50k (BitFlipping) / 20k (Fetch) |
| 3.7 | 4 | `lab_3_7_optuna.py` | Búsqueda con Optuna y reentrenamiento con semillas nuevas | 20 trials × 5 tramos de 40k |

Cada script admite `--help`. Opciones comunes: `--seeds 0,1,2` o `--seeds 0-4`, `--steps`, `--eval-every`, `--device`.

### Comandos de cada sesión

```bash
# Semana 1
python experiments_3/lab_3_1_dqn.py --env CartPole-v1 --config default
python experiments_3/lab_3_1_dqn.py --env CartPole-v1 --config zoo
python experiments_3/lab_3_1_dqn.py --env LunarLander-v3 --config slides --seeds 0,1,2
python experiments_3/lab_3_2_ablacion.py
python experiments_3/plot_results.py logs/lab_3_2

# Semana 2 (entregable): ver la sección «Implementación evaluable»
python experiments_3/snake_env.py --human                      # ver el entorno
python experiments_3/lab_3_3_reinforce.py --check
python experiments_3/lab_3_3_reinforce.py --obs original --seeds 0,1 --steps 100000 --tag it01_original
python experiments_3/diario.py logs/lab_3_3
python experiments_3/lab_3_3_reinforce.py --obs original --watch logs/lab_3_3/it01_original/seed_0/policy.pt

# Semana 3
python experiments_3/lab_3_4_ppo_a2c.py
python experiments_3/lab_3_4_ppo_a2c.py --algos ppo --clip 0.1,0.2,0.4
python experiments_3/lab_3_5_continuo.py --env Pendulum-v1 --algos ddpg,td3,sac
python experiments_3/lab_3_5_continuo.py --env Pendulum-v1 --algos td3 --sigma 0.0,0.1,0.5
python experiments_3/lab_3_5_continuo.py --env ShowerContinuous-v0 --algos sac,ppo --steps 30000

# Semana 4
python experiments_3/lab_3_6_her.py bitflip
python experiments_3/lab_3_6_her.py fetch
python experiments_3/plot_results.py logs/lab_3_6_FetchReach-v4 --metric success
python experiments_3/lab_3_7_optuna.py search --trials 20
python experiments_3/lab_3_7_optuna.py final --seeds 100-104
python experiments_3/plot_results.py logs/lab_3_7 logs/lab_3_4 logs/lab_3_1 --configs ppo_tuned,ppo_default,a2c_default,LunarLander-v3_slides
```

## Implementación evaluable (15 %): mejorar un agente de REINFORCE en el Snake

No se evalúa cuántas manzanas come vuestro agente, sino cómo llegáis a él: qué probáis, por qué, qué esperabais que pasara y qué decidís con lo que sale.

**1. Implementación (requisito).** Completad los TODO 1–5 de `lab_3_3_reinforce.py`: red de política, `reward_to_go`, `collect_episode`, `policy_loss` y la versión con baseline (`ValueNet`). Debe pasar `--check` y aprender en CartPole (`--env CartPole-v1 --steps 50000`). Podéis usar asistentes de IA: indicad en el diario para qué. PyTorch y Gymnasium sí; implementaciones ya hechas del algoritmo, no.

**2. Diario de iteraciones (el núcleo de la entrega).** Mínimo 6 iteraciones en `DIARIO.md` (hay una plantilla). Cada una cambia una sola cosa y se escribe antes de lanzarla:

- **Hipótesis**: qué creéis que limita al agente y por qué.
- **Cambio**: una sola variable.
- **Predicción**: qué esperáis que pase con las manzanas, escrita antes de ver resultados.
- **Resultado**: la fila de `diario.py`, que se genera a partir de los logs.
- **Decisión**: se queda, se descarta o qué se prueba después.

Iteraciones obligatorias: REINFORCE con la observación original (punto de partida); al menos una observación propia (TODO 6); una ablación de un rasgo de vuestra observación; con y sin baseline; comparación con A2C y PPO (`lab_3_3_referencia_sb3.py`). El resto, libres. Una iteración que no mejora, bien analizada, vale lo mismo que una que mejora.

**3. Explorar barato, confirmar caro.** Exploración: 2 semillas y 100 000 pasos. Confirmación: las 2–3 configuraciones finales, con 5 semillas y 200 000 pasos. Justificad qué confirmáis.

**4. Diagnóstico.** Antes de cambiar nada, mirad cómo acaban las partidas (pared, cuerpo u hambre: columnas de `diario.py`) y ved jugar al agente (`--watch`, o `--gif` en Colab).

**5. Reglas de diseño (TODO 6 y 7, en `snake_design.py`).** La observación describe el estado: no puede contener acciones recomendadas ni el resultado de una búsqueda (BFS, A*, heurísticas). Los rasgos calculados, como las casillas libres en cada dirección, se permiten si se declaran en el diario. La recompensa puede quedarse en `sparse` si se justifica; cambiarla es una iteración más.

**6. Qué se entrega.** Repositorio git (o zip con la carpeta `.git`) con un commit por iteración, que incluya: el código, `DIARIO.md`, la carpeta `logs/lab_3_3` completa (`config.json` y `evaluations.npz`; de los modelos `.pt`, solo el final) y un informe en PDF de 4 páginas como máximo.

**7. Defensa (5 minutos).** El profesor elige una iteración de vuestro diario y pregunta por la decisión que tomasteis. Después os pide un cambio: predecís su efecto, lo ejecutáis y lo comentáis. Si no se puede defender el diario, los criterios de proceso y diagnóstico quedan en el nivel 1.

### Comandos

```bash
python experiments_3/lab_3_3_reinforce.py --check                       # tras cada TODO

# exploración: una carpeta por iteración con --tag
python experiments_3/lab_3_3_reinforce.py --obs original --seeds 0,1 --steps 100000 --tag it01_original
python experiments_3/lab_3_3_reinforce.py --obs custom --seeds 0,1 --steps 100000 --tag it02_peligro

# confirmación de las configuraciones finales
python experiments_3/lab_3_3_reinforce.py --obs custom --seeds 0-4 --tag it07_final
python experiments_3/lab_3_3_reinforce.py --obs custom --baseline --seeds 0-4 --tag it08_final_baseline
python experiments_3/lab_3_3_referencia_sb3.py --obs custom --seeds 0-4 --tag it09_ref

python experiments_3/diario.py logs/lab_3_3                             # tabla para DIARIO.md
python experiments_3/plot_results.py logs/lab_3_3 --metric score --configs it07_final,it08_final_baseline
```

El rendimiento se mide en **manzanas por partida**, no en retorno: con recompensas distintas, los retornos no son comparables. `diario.py` indica en la última columna qué cambió respecto a la iteración anterior (incluido el código de `SnakeObservation` y `SnakeReward`); si aparece más de una cosa, la iteración no cambió una sola variable.

Orientación (no se puntúa): una política aleatoria come 0,04 manzanas por partida; REINFORCE con la observación original, 1–2; con una buena observación, más de 10.

### Rúbrica

| Criterio | Peso | 1 · Insuficiente | 2 · Adecuado | 3 · Excelente |
|---|---|---|---|---|
| Implementación | 10 % | No pasa `--check` (requisito: sin esto no se corrige) | Pasa `--check` y aprende en CartPole | Además, decisiones de implementación justificadas (normalización, tamaño del lote) |
| Proceso | 35 % | Cambios sin hipótesis, varias variables a la vez o predicciones escritas después | Al menos 6 iteraciones completas, una variable cada vez | Cadena de decisiones coherente; las iteraciones fallidas se aprovechan para la siguiente |
| Diagnóstico | 20 % | Cambios a ciegas | Usa las causas de fin de partida y mira jugar al agente | Identifica el cuello de botella y lo ataca con un cambio medido |
| Conclusiones | 15 % | Afirmaciones sin datos o con una sola semilla | Baseline y A2C/PPO comparados con 5 semillas e intervalo | Explica las diferencias y los límites de su diseño |
| Defensa | 20 % | No sabe explicar su diario | Explica sus decisiones | Predice bien el efecto del cambio en directo y lo razona |

## Notas de versiones

- Gymnasium 1.x: `LunarLander-v3`, `CartPole-v1`, `Pendulum-v1`.
- Gymnasium-Robotics con los bindings actuales de MuJoCo: `FetchReach-v4`, `FetchPush-v4`. Con versiones antiguas del paquete el sufijo puede ser `-v3`; comprobadlo con `gym.pprint_registry()`.
- El DQN de SB3 es la versión básica (sin Double, Dueling ni PER); QR-DQN y TQC están en `sb3-contrib`.
- Los valores marcados como «basados en rl_zoo3» conviene contrastarlos con los YAML de la versión instalada (`hyperparams/*.yml`).
