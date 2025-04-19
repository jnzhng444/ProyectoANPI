import sympy as sp
import numpy as np
import time

# Constantes físicas
e = 1.602e-19       # Carga elemental (C)
k = 1.38e-23        # Constante de Boltzmann (J/K)
T = 298             # Temperatura ambiente (K)
epsilon_0 = 8.854e-12  # F/m
epsilon_r = 78.5       # Agua a 25°C
epsilon = epsilon_r * epsilon_0

# Parámetros del modelo
rho_0 = 1e3 * e     # C/m³, densidad de carga típica
h = 1e-9            # paso de malla pequeño (1 nm)

# Constantes A y B
A_val = (rho_0 * h**2) / epsilon
B_val = e / (k * T)

# Reescribimos la ecuación como f(phi) = phi - A * exp(-B * phi)
phi = sp.symbols('x')  # Usamos x
f_expr = phi - A_val * sp.exp(-B_val * phi)

#Se convierte f_expr a string
f_str = str(f_expr)

#Metodo biseccion
def biseccion(f1, a, b, tol, iterMax):
    x = sp.symbols('x')  # Declara una variable simbólica 'x'.
    f = sp.lambdify(x, f1, 'numpy')  # Convierte la función simbólica a una función numérica compatible con NumPy.

    # Verifica que exista un cambio de signo en el intervalo (Teorema de Bolzano).
    if f(a) * f(b) < 0:
        start_time = time.perf_counter()  # Inicia el contador de tiempo.

        for k in range(1, iterMax + 1):
            x_ = (a + b) / 2  # Calcula el punto medio del intervalo.

            # Selecciona el subintervalo que contiene la raíz.
            if f(a) * f(x_) < 0:
                b = x_
            else:
                a = x_

            error = abs(f(x_))  # Calcula el error actual (valor absoluto de f(x_)).

            # Si el error es menor que la tolerancia, retorna la solución.
            if error < tol:
                end_time = time.perf_counter()  # Finaliza el contador.
                return x_, error, k, end_time - start_time

        # Si no converge en iterMax iteraciones, retorna None.
        print(f"No se alcanzó la tolerancia en {iterMax} iteraciones.")
        return None, None, None, None
    else:
        print("No cumple el teorema de Bolzano")  # Si no hay cambio de signo.
        return None, None, None, None


# Newton-Raphson
def newton_raphson(f, x0, tol, iterMax):
    x = sp.symbols('x')  # Declara variable simbólica.
    fs = sp.sympify(f)  # Interpreta la función f como expresión simbólica.
    fsp = sp.diff(fs, x)  # Calcula la derivada de la función.

    # Convierte función y derivada a versión numérica.
    fn = sp.lambdify(x, fs, 'numpy')
    fnp = sp.lambdify(x, fsp, 'numpy')

    xk = x0  # Inicializa la variable con la primera aproximación.
    start_time = time.perf_counter()

    for k in range(1, iterMax + 1):
        if fnp(xk) == 0:
            raise ValueError("La derivada es cero.")  # Evita división por cero.

        xk_new = xk - fn(xk) / fnp(xk)  # Aplica la fórmula de Newton-Raphson.
        error = max(abs(fn(xk_new)), abs(xk_new - xk))  # Calcula el error.

        if error < tol:
            end_time = time.perf_counter()
            return xk_new, error, k, end_time - start_time

        xk = xk_new  # Actualiza la variable para la próxima iteración.

    end_time = time.perf_counter()
    return xk, error, iterMax, end_time - start_time


# Método de la Secante
def secante(f, x0, x1, tol, iterMax):
    x = sp.symbols('x') # Declara una variable simbólica.
    fs = sp.sympify(f) # Convierte la función f (string) a expresión simbólica.
    fn = sp.lambdify(x, fs, 'numpy') # Convierte la función simbólica a función numérica con NumPy.

    start_time = time.perf_counter() # Inicia el cronómetro de alta precisión.

    for k in range(1, iterMax + 1):
        f_x0 = fn(x0) # Evalúa la función en x0.
        f_x1 = fn(x1) # Evalúa la función en x1.
        denominator = f_x1 - f_x0 # Calcula el denominador de la fórmula de la secante.

        if denominator == 0: # Evita división por cero.
            raise ValueError(f"División por cero en la iteración {k}.")

        xk_new = x1 - (f_x1 * (x1 - x0)) / denominator # Aplica la fórmula de la secante.
        error = max(abs(fn(xk_new)), abs(xk_new - x1)) # Calcula el error como máximo entre f(x) y |x_n+1 - x_n|.

        if error < tol: # Si el error es suficientemente pequeño, termina.
            end_time = time.perf_counter()
            return xk_new, error, k, end_time - start_time

        x0 = x1 # Actualiza x0 y x1 para la siguiente iteración.
        x1 = xk_new

    end_time = time.perf_counter() # Finaliza el cronómetro si se alcanza el límite de iteraciones.
    return xk_new, error, iterMax, end_time - start_time


# Halley
def halley_method(f, x0, tol, iterMax):
    x = sp.symbols('x')
    fs = sp.sympify(f) # Convierte la función a expresión simbólica.
    fsp = sp.diff(fs, x) # Calcula la primera derivada.
    fspp = sp.diff(fsp, x) # Calcula la segunda derivada.

    # Convierte las funciones simbólicas a funciones numéricas.
    fn = sp.lambdify(x, fs, 'numpy')
    fnp = sp.lambdify(x, fsp, 'numpy')
    fnpp = sp.lambdify(x, fspp, 'numpy')

    xk = x0  # Inicializa la variable con la primera aproximación.
    start_time = time.perf_counter()

    for k in range(1, iterMax + 1):
        if fnp(xk) == 0: # Verifica que la derivada no sea cero.
            raise ValueError("Derivada cero.")

        # Aplica la fórmula del método de Halley.
        xk_new = xk - (fn(xk) / fnp(xk)) * (1 / (1 - (fn(xk) * fnpp(xk)) / (2 * fnp(xk)**2)))
        error = max(abs(fn(xk_new)), abs(xk_new - xk))

        if error < tol:
            end_time = time.perf_counter()
            return xk_new, error, k, end_time - start_time

        xk = xk_new # Actualiza xk para la siguiente iteración.

    end_time = time.perf_counter()
    return xk, error, iterMax, end_time - start_time


# Dekker
def dekker_method(f, a, b, tol, iterMax):
    x = sp.symbols('x')
    fs = sp.sympify(f) # Convierte la función de entrada a expresión simbólica.
    fn = sp.lambdify(x, fs, 'numpy') # Función numérica.

    fa = fn(a)
    fb = fn(b)

    if fa * fb >= 0:# Verifica que exista un cambio de signo (Bolzano).
        raise ValueError("No hay cambio de signo.")

    start_time = time.perf_counter()

    for k in range(1, iterMax + 1):
        s = b - fb * (b - a) / (fb - fa)# Intenta usar el paso de la secante.

        if a < s < b:
            fs_val = fn(s)# Evalúa f en s.
            if abs(fs_val) < tol: # Si el error es pequeño, termina.
                end_time = time.perf_counter()
                return s, abs(fs_val), k, end_time - start_time
        else:
            s = (a + b) / 2 # Si el paso de secante no es válido, hace bisección.
            fs_val = fn(s)

        if fa * fs_val < 0:# Actualiza el intervalo dependiendo del signo.
            b, fb = s, fs_val
        else:
            a, fa = s, fs_val

    end_time = time.perf_counter()
    return s, abs(fs_val), iterMax, end_time - start_time

# Chebyshev
def chebyshev_method(f, x0, tol, iterMax):
    x = sp.symbols('x')
    #Calculo de derivadas
    fs = sp.sympify(f)
    fsp = sp.diff(fs, x)
    fspp = sp.diff(fsp, x)
    #Se pasa a numerico
    fn = sp.lambdify(x, fs, 'numpy')
    fnp = sp.lambdify(x, fsp, 'numpy')
    fnpp = sp.lambdify(x, fspp, 'numpy')
    xk = x0
    start_time = time.perf_counter()
    for k in range(1, iterMax + 1):
        fx, fpx, fppx = fn(xk), fnp(xk), fnpp(xk)
        if fpx == 0: #Verifica que no sea cero
            raise ValueError("Derivada cero.")
        xk_new = xk - (fx / fpx) * (1 + (fx * fppx) / (2 * fpx**2)) # Se calcula la nueva aproximación
        error = max(abs(fn(xk_new)), abs(xk_new - xk))  # Se calcula el error entre iteraciones
        if error < tol:  # Si se cumple el criterio de tolerancia
            end_time = time.perf_counter() # Se detiene el cronómetro
            return xk_new, error, k, end_time - start_time # Se retorna la raíz, error, iteraciones y tiempo
        xk = xk_new # Se actualiza xk para la siguiente iteración
    end_time = time.perf_counter() # Si no converge, se detiene el cronómetro al final del bucle
    return xk, error, iterMax, end_time - start_time


# Parámetros iniciales para todos los métodos
x0 = 0.1  # Primera aproximación
x1 = 0.5  # Segunda aproximación
a = -10  # Para el método de bisección y dekker
b = 20  # Para el método de bisección y dekker
tol = 1e-15  # Tolerancia
iterMax = 20000  # Número máximo de iteraciones

# Ejecución de los métodos y mostrar resultados
print("Método de Bisección:")
x_biseccion, error_biseccion, k_biseccion, exec_time_biseccion = biseccion(f_str, a, b, tol, iterMax)
print(f"Valores iniciales: a = {a}, b = {b}")
print(f"Aproximación xk: {x_biseccion}")
print(f"Error ek: {error_biseccion}")
print(f"Iteraciones k: {k_biseccion}")
print(f"Tiempo de ejecución: {exec_time_biseccion * 1e6:.3f} microsegundos\n")


print("Método de Newton-Raphson:")
x_newton, error_newton, k_newton, exec_time_newton = newton_raphson(f_str, x0, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}")
print(f"Aproximación xk: {x_newton}")
print(f"Error ek: {error_newton}")
print(f"Iteraciones k: {k_newton}")
print(f"Tiempo de ejecución: {exec_time_newton * 1e6:.3f} microsegundos\n")

print("Método de la Secante:")
root_secante, error_secante, num_iterations_secante, exec_time_secante = secante(f_str, x0, x1, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}, x1 = {x1}")
print(f"Aproximación xk: {root_secante}")
print(f"Error ek: {error_secante}")
print(f"Iteraciones k: {num_iterations_secante}")
print(f"Tiempo de ejecución: {exec_time_secante * 1e6:.3f} microsegundos\n")

print("Método de Halley:")
x_halley, error_halley, k_halley, exec_time_halley = halley_method(f_str, x0, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}")
print(f"Aproximación xk: {x_halley}")
print(f"Error ek: {error_halley}")
print(f"Iteraciones k: {k_halley}")
print(f"Tiempo de ejecución: {exec_time_halley * 1e6:.3f} microsegundos\n")

print("Método de Dekker:")
root_dekker, error_dekker, iterations_dekker, exec_time_dekker = dekker_method(f_str, a, b, tol, iterMax)
print(f"Valores iniciales: a = {a}, b = {b}")
print(f"Aproximación xk: {root_dekker}")
print(f"Error ek: {error_dekker}")
print(f"Iteraciones k: {iterations_dekker}")
print(f"Tiempo de ejecución: {exec_time_dekker * 1e6:.3f} microsegundos\n")

print("Método de Chebyshev:")
x_chebyshev, error_chebyshev, k_chebyshev, exec_time_chebyshev = chebyshev_method(f_str, x0, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}")
print(f"Aproximación xk: {x_chebyshev}")
print(f"Error ek: {error_chebyshev}")
print(f"Iteraciones k: {k_chebyshev}")
print(f"Tiempo de ejecución: {exec_time_chebyshev * 1e6:.3f} microsegundos\n")

