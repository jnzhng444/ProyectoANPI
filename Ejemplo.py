import sympy as sp
import numpy as np
import time

# IMPORTACIÓN DE MÉTODOS DESDE OTRO ARCHIVO
# Asumimos que tu archivo original se llama `metodos_iterativos.py`
from IterativeMethods import (
    biseccion,
    newton_raphson,
    secante,
    halley_method,
    dekker_method,
    chebyshev_method
)

# -------------------------------
# DEFINICIÓN DE LA NUEVA FUNCIÓN
# -------------------------------
x = sp.symbols('x')
f_expr = x * sp.exp(x) - 1     # expresión simbólica
f_str = str(f_expr)

# -------------------------------
# PARÁMETROS DE EJEMPLO
# -------------------------------
x0 = 0.5
x1 = 1.0
a = 0
b = 1
tol = 1e-15
iterMax = 20000

# -------------------------------
# APLICACIÓN DE LOS MÉTODOS
# -------------------------------
print("Método de Bisección:")
x_bis, e_bis, k_bis, t_bis = biseccion(f_expr, a, b, tol, iterMax)
print(f"xk = {x_bis}, error = {e_bis}, iter = {k_bis}, tiempo = {t_bis*1e6:.3f} µs\n")

print("Método de Newton-Raphson:")
x_newton, e_newton, k_newton, t_newton = newton_raphson(f_str, x0, tol, iterMax)
print(f"xk = {x_newton}, error = {e_newton}, iter = {k_newton}, tiempo = {t_newton*1e6:.3f} µs\n")

print("Método de la Secante:")
x_sec, e_sec, k_sec, t_sec = secante(f_str, x0, x1, tol, iterMax)
print(f"xk = {x_sec}, error = {e_sec}, iter = {k_sec}, tiempo = {t_sec*1e6:.3f} µs\n")

print("Método de Halley:")
x_hal, e_hal, k_hal, t_hal = halley_method(f_str, x0, tol, iterMax)
print(f"xk = {x_hal}, error = {e_hal}, iter = {k_hal}, tiempo = {t_hal*1e6:.3f} µs\n")

print("Método de Dekker:")
x_dek, e_dek, k_dek, t_dek = dekker_method(f_str, a, b, tol, iterMax)
print(f"xk = {x_dek}, error = {e_dek}, iter = {k_dek}, tiempo = {t_dek*1e6:.3f} µs\n")

print("Método de Chebyshev:")
x_cheb, e_cheb, k_cheb, t_cheb = chebyshev_method(f_str, x0, tol, iterMax)
print(f"xk = {x_cheb}, error = {e_cheb}, iter = {k_cheb}, tiempo = {t_cheb*1e6:.3f} µs\n")
