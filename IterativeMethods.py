import sympy as sp
import numpy as np
import time

# Método de Bisección
def biseccion(f1, a, b, tol, iterMax):
    # Convierte la cadena de texto de la función en una función simbólica
    x = sp.symbols('x')
    f = sp.lambdify(x, f1, 'numpy')

    # Comprobamos si el teorema de Bolzano se cumple (es decir, si hay un cambio de signo)
    if f(a) * f(b) < 0:
        start_time = time.time()  # Iniciar tiempo de ejecución
        for k in range(1, iterMax + 1):
            # Paso 2: Calcula el punto medio
            x_ = (a + b) / 2
            # Paso 3: Selecciona el subintervalo
            if f(a) * f(x_) < 0:
                b = x_
            else:
                a = x_

            # Calcular el error
            error = abs(f(x_))

            # Paso 4: Si el error es menor que la tolerancia, se detiene
            if error < tol:
                end_time = time.time()  # Finalizar tiempo de ejecución
                execution_time = end_time - start_time
                return x_, error, k, execution_time

        else:
            # Si no se alcanza la tolerancia después de iterMax iteraciones
            print(f"No se alcanzó la tolerancia en {iterMax} iteraciones.")
            return None, None, None, None
    else:
        print("No cumple el teorema de Bolzano")
        return None, None, None, None


# Método de Newton-Raphson
def newton_raphson(f, x0, tol, iterMax):
    # Definir la variable simbólica
    x = sp.symbols('x')

    # Función simbólica
    fs = sp.sympify(f)

    # Derivada simbólica de la función
    fsp = sp.diff(fs, x)

    # Convertir las funciones simbólicas a funciones numéricas
    fn = sp.lambdify(x, fs, 'numpy')
    fnp = sp.lambdify(x, fsp, 'numpy')

    # Inicializar el valor de xk
    xk = x0

    start_time = time.time()  # Iniciar tiempo de ejecución

    # Realizar el método de Newton-Raphson
    for k in range(1, iterMax + 1):
        if fnp(xk) == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")

        # Actualización de xk
        xk_new = xk - fn(xk) / fnp(xk)

        # Calcular el error
        error = max(abs(fn(xk_new)), abs(xk_new - xk))

        # Si el error es menor que la tolerancia, detener el bucle
        if error < tol:
            end_time = time.time()  # Finalizar tiempo de ejecución
            execution_time = end_time - start_time
            return xk_new, error, k, execution_time

        xk = xk_new

    end_time = time.time()  # Finalizar tiempo de ejecución
    execution_time = end_time - start_time
    return xk, error, iterMax, execution_time

def secante(f, x0, x1, tol, iterMax):
    # Definir la variable simbólica
    x = sp.symbols('x')

    # Función simbólica
    fs = sp.sympify(f)

    # Convertir la función simbólica a función numérica
    fn = sp.lambdify(x, fs, 'numpy')

    # Inicializar los valores iniciales de x0, x1
    start_time = time.time()  # Iniciar tiempo de ejecución

    for k in range(1, iterMax + 1):
        # Evaluar la función en x0 y x1
        f_x0 = fn(x0)
        f_x1 = fn(x1)

        # Evitar la división por cero
        denominator = f_x1 - f_x0
        if denominator == 0:
            raise ValueError(f"División por cero detectada en la iteración {k}.")

        # Aplicar la fórmula del método de la secante
        xk_new = x1 - (f_x1 * (x1 - x0)) / denominator

        # Calcular el error
        error = max(abs(fn(xk_new)), abs(xk_new - x1))

        # Comprobar si el error es menor que la tolerancia
        if error < tol:
            end_time = time.time()  # Finalizar tiempo de ejecución
            execution_time = end_time - start_time
            return xk_new, error, k, execution_time

        # Actualizar las aproximaciones para la siguiente iteración
        x0 = x1
        x1 = xk_new

    end_time = time.time()  # Finalizar tiempo de ejecución
    execution_time = end_time - start_time
    return xk_new, error, iterMax, execution_time


def halley_method(f, x0, tol, iterMax):
    # Definir la variable simbólica
    x = sp.symbols('x')

    # Función simbólica
    fs = sp.sympify(f)

    # Derivadas simbólicas de la función (primera y segunda derivada)
    fsp = sp.diff(fs, x)
    fspp = sp.diff(fsp, x)

    # Convertir las funciones simbólicas a funciones numéricas
    fn = sp.lambdify(x, fs, 'numpy')
    fnp = sp.lambdify(x, fsp, 'numpy')
    fnpp = sp.lambdify(x, fspp, 'numpy')

    # Inicializar el valor de xk
    xk = x0

    start_time = time.time()  # Iniciar tiempo de ejecución

    # Realizar el método de Halley (fórmula simplificada)
    for k in range(1, iterMax + 1):
        if fnp(xk) == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")

        # Actualización de xk usando la fórmula simplificada de Halley
        xk_new = xk - (fn(xk) / fnp(xk)) * (1 / (1 - (fn(xk) * fnpp(xk)) / (2 * (fnp(xk)) ** 2)))

        # Calcular el error
        error = max(abs(fn(xk_new)), abs(xk_new - xk))

        # Si el error es menor que la tolerancia, detener el bucle
        if error < tol:
            end_time = time.time()  # Finalizar tiempo de ejecución
            execution_time = end_time - start_time
            return xk_new, error, k, execution_time

        xk = xk_new

    end_time = time.time()  # Finalizar tiempo de ejecución
    execution_time = end_time - start_time
    return xk, error, iterMax, execution_time


def dekker_method(f, a, b, tol, iterMax):
    # Definir la variable simbólica
    x = sp.symbols('x')

    # Función simbólica
    fs = sp.sympify(f)

    # Derivada simbólica de la función
    fsp = sp.diff(fs, x)

    # Convertir las funciones simbólicas a funciones numéricas
    fn = sp.lambdify(x, fs, 'numpy')

    # Inicializar el valor de a y b
    fa = fn(a)
    fb = fn(b)

    if fa * fb >= 0:
        raise ValueError("No hay cambio de signo en el intervalo [a, b]. El método no puede continuar.")

    start_time = time.time()  # Iniciar tiempo de ejecución

    # Realizar el método de Dekker
    for k in range(1, iterMax + 1):
        # Paso de la secante
        s = b - fb * (b - a) / (fb - fa)

        # Si la secante es válida y está en el intervalo
        if a < s < b:
            fs = fn(s)
            # Si el error es menor que la tolerancia, detener el bucle
            if abs(fs) < tol:
                end_time = time.time()  # Finalizar tiempo de ejecución
                execution_time = end_time - start_time
                return s, abs(fs), k, execution_time

        # Si la secante no es válida, usar bisección
        else:
            s = (a + b) / 2
            fs = fn(s)

        # Actualizar el intervalo
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

    end_time = time.time()  # Finalizar tiempo de ejecución
    execution_time = end_time - start_time
    return s, abs(fs), iterMax, execution_time

# Parámetros iniciales para todos los métodos
x0 = 1.0  # Primera aproximación
x1 = 2.0  # Segunda aproximación
a = 1  # Para el método de bisección
b = 3  # Para el método de bisección
tol = 1e-6  # Tolerancia
iterMax = 20000  # Número máximo de iteraciones

# Ejecución de los métodos y mostrar resultados
print("Método de Bisección:")
x_biseccion, error_biseccion, k_biseccion, exec_time_biseccion = biseccion('x**2 - 4', a, b, tol, iterMax)
print(f"Valores iniciales: a = {a}, b = {b}")
print(f"Aproximación xk: {x_biseccion}")
print(f"Error ek: {error_biseccion}")
print(f"Iteraciones k: {k_biseccion}")
print(f"Tiempo de ejecución: {exec_time_biseccion:.6f} segundos\n")

print("Método de Newton-Raphson:")
x_newton, error_newton, k_newton, exec_time_newton = newton_raphson('x**2 - 4', x0, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}")
print(f"Aproximación xk: {x_newton}")
print(f"Error ek: {error_newton}")
print(f"Iteraciones k: {k_newton}")
print(f"Tiempo de ejecución: {exec_time_newton:.6f} segundos\n")

print("Método de la Secante:")
root_secante, num_iterations_secante, error_secante, exec_time_secante = secante('x**2 - 4', x0, x1, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}, x1 = {x1}")
print(f"Aproximación xk: {root_secante}")
print(f"Error ek: {error_secante}")
print(f"Iteraciones k: {num_iterations_secante}")
print(f"Tiempo de ejecución: {exec_time_secante:.6f} segundos\n")

print("Método de Halley:")
x_halley, error_halley, k_halley, exec_time_halley = halley_method('x**2 - 4', x0, tol, iterMax)
print(f"Valores iniciales: x0 = {x0}")
print(f"Aproximación xk: {x_halley}")
print(f"Error ek: {error_halley}")
print(f"Iteraciones k: {k_halley}")
print(f"Tiempo de ejecución: {exec_time_halley:.6f} segundos\n")

print("Método de Dekker:")
root_dekker, error_dekker, iterations_dekker, exec_time_dekker = dekker_method('x**2 -4', a, b, tol, iterMax)
print(f"Valores iniciales: a = {a}, b = {b}")
print(f"Aproximación xk: {root_dekker}")
print(f"Error ek: {error_dekker}")
print(f"Iteraciones k: {iterations_dekker}")
print(f"Tiempo de ejecución: {exec_time_dekker:.6f} segundos\n")
