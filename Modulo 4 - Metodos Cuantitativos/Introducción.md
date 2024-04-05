---
title: Introducción
author: Raúl Monroy
tags:
  - "#School"
  - Introduccion
  - Lecture
  - AI
  - Compiladores
  - Metodos_Cuantitativos
---

Modelo = Es una herramienta

De cierta forma a mayor abstaccion existe un menor detalle del modelo  y viceversa. 

## Modelo 
Un modelo es un objeto que representa a un sistema y que permite predecir su comportamiento;

- Grado de 
-  Una vez definido, un modelo puede
	-  *Parametrizarse*, para caracterizar alternativas por considerarse;

## Evaluación experimental
- No experimentamos directamente con el sistema, porque:
	-  Afecta su *funcionalidad*:
		-  Corta la continuidad de un proceso;
		- Incurre en riesgos
	-  El elemento de interes puede no ser *accesible*;
	- Es incosteable o tardo;
	- Es dificil generalizar los resultados obtenidos de experimentar en una configuración a otra;
	- Es imposble cuando el sistema no existe aun.


### Leyes operacionales:

Leyes simples que no se basan en consideraciones sobre distribuciones de Tiempo de servicio o de llegada, resultan de aplicar álgebra  ́elemental (tipo f = ma), se les denominan “leyes” porque se cumplen sin Consideracion alguna.  ́

*Principios (consideraciones) operacionales:* 
-  Las propiedades son medibles precisamente (y pueden validarse directamente);
-  El sistema satisface la propiedad de balance de flujo (llegadas=salidas); 
	- conservacion de trabajo ;
- Los dispositivos son homogeneos 
	-  Tiempo que toma a un dispositivo satisfacer una peticion,  Llamado tiempo medio de servicio, es independiente del estado del sistema;
	-  Consideramos que no hay interbloqueos o sincronizacion. 

B_k  (Busy time):
	Tiempo total en T que el d_k estuvo en uso (ocupado) 
C_r: 
	# de "unidades de trabajo del dispositvio k" completado por él  mismo 


![](https://i.imgur.com/zrfHQbs.png)


## Aspectos de la ley de Little
- Relacion entre la  ́ ley de Little y la ley de utilizacion.
![](https://i.imgur.com/j6E7TjA.png)

- La ley de Little puede aplicarse a muchos niveles diferentes:
	-  componente (con o sin la cola de peticiones)
	-  subsistema, y 
	-  sistema



### Aplicacion a nivel componente, parte I 
Nivel componente: numero medio de peticiones en el
Componente = utilizacion.

### Aplicacion a nivel componente, parte II
Nivel componente, incluyendo cola de peticiones:
	- Poblacion: # total de peticiones de servicio o en espera.
	-  Tiempo medio de residencia = tiempo medio de espera mas tiempo medio de servicio

![](https://i.imgur.com/1Ukd10e.png)
