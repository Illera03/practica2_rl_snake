| Iteración | Algoritmo | Obs | Recompensa | Semillas | Pasos | Manzanas | IC 95 % | IQM | Pared | Cuerpo | Hambre | Cambia respecto a la anterior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| it01_original | REINFORCE | original | sparse | 2 | 100.013 | 0,16 | [0,07, 0,25] | 0,16 | 100 % | 0 % | 0 % | punto de partida |
| it02_peligro | REINFORCE | custom | sparse | 2 | 100.561 | 0,42 | [0,17, 0,68] | 0,42 | 0 % | 0 % | 100 % | obs |
| it03_comida | REINFORCE | custom | sparse | 2 | 100.026 | 20,12 | [17,85, 22,40] | 20,12 | 0 % | 100 % | 0 % | código de la observación |
| it04_espacio | REINFORCE | custom | sparse | 2 | 100.002 | 1,48 | [0,57, 2,38] | 1,48 | 36 % | 18 % | 47 % | código de la observación |
| it05_sin_peligro | REINFORCE | custom | sparse | 2 | 100.001 | 0,20 | [0,20, 0,20] | 0,20 | 100 % | 0 % | 0 % | código de la observación |
