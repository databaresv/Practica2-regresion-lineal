# Práctica 2 — Programación Orientada a Objetos
Framework de regresión lineal múltiple en C++

**Curso:** Lenguajes y Paradigmas de Programación — EAFIT, 2026-2
**Autor:** Daniel Alejandro Tabares Vasquez (databaresv@eafit.edu.co)

## Descripción
Clase `LinearRegression` que entrena un modelo de regresión lineal (simple y múltiple)
con gradiente descendente, sin librerías externas. Incluye los métodos `fit`, `predict`,
`score`, `data_scaling` y `r2_score`, un análisis de memoria (global, local, dinámica,
vector y arreglo dinámico) y pruebas de validación de errores.

## Archivos
- `main.cpp`: código fuente completo.
- `ice_cream.txt`: datos para la regresión simple (temperatura vs. ventas de helados).
- `student_scores.txt`: datos para la regresión múltiple (calificaciones de estudiantes).
- `Informe Técnico — Práctica 2.1.pdf`: informe de la práctica.

Los datos son los CSV de EAFIT Interactiva guardados con extensión .txt, porque
OnlineGDB trata los archivos .csv como scripts de enlazado.

## Cómo compilar y ejecutar
Con los dos archivos .txt en la misma carpeta que main.cpp:

    g++ -std=c++17 main.cpp -o main
    ./main

También se puede ejecutar en OnlineGDB subiendo los tres archivos al proyecto.
