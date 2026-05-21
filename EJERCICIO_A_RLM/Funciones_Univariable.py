# ===========================================================================
# FUNCIONES BASE
# ===========================================================================

def predecir(x, w, b):
    return w * x + b

def calcular_mse(datos, w, b):
    n = len(datos)
    total = sum((y - predecir(x, w, b))**2 for x, y in datos)
    return total / n

def calcular_mae(datos, w, b):
    n = len(datos)
    total = sum(abs(y - predecir(x, w, b)) for x, y in datos)
    return total / n

def gradiente_mse(datos, w, b):
    n = len(datos)
    grad_w = 0.0
    grad_b = 0.0
    for x, y in datos:
        error = y - predecir(x, w, b)
        grad_w += -2 * x * error
        grad_b += -2 * error
    return grad_w / n, grad_b / n

def gradiente_una_muestra(x, y, w, b):
    error = y - predecir(x, w, b)
    grad_w = -2 * x * error
    grad_b = -2 * error
    return grad_w, grad_b

def solucion_analitica(datos):
    n = len(datos)
    sx = sum(x for x, y in datos)
    sy = sum(y for x, y in datos)
    sxy = sum(x*y for x, y in datos)
    sx2 = sum(x**2 for x, y in datos)

    m = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b = (sy - m * sx) / n
    return m, b

def separador(titulo):
    linea = "=" * 60
    print(f"\n{linea}")
    print(f" {titulo}")
    print(linea)

# ===========================================================================
# GRADIENT DESCENT (GD)
# ===========================================================================

def gradient_descent(datos, lr=0.01, iteraciones=1000, verbose=True, mostrar_cada=100):
    w, b = 0.0, 0.0
    historial = []

    if verbose:
        separador("GRADIENT DESCENT (GD)")
        print(f" lr={lr} | N={len(datos)} | iter_max={iteraciones}")
        print(f" {'Iter':>6} {'w':>10} {'b':>10} {'MSE':>12}")
        print(f" {'-'*6} {'-'*10} {'-'*10} {'-'*12}")

    for i in range(iteraciones):
        mse = calcular_mse(datos, w, b)
        historial.append((i, w, b, mse))

        # calcular gradiente con todos los datos
        gw, gb = gradiente_mse(datos, w, b)

        # actualizar parametros
        w = w - lr * gw
        b = b - lr * gb

    mse_final = calcular_mse(datos, w, b)
    historial.append((iteraciones, w, b, mse_final))

    if verbose:
        print(f" {iteraciones:>6} {w:>10.4f} {b:>10.4f} {mse_final:>12.4f}")
        print(f"\n Resultado GD: w = {w:.4f} | b = {b:.4f}")
        print(f" MSE final : {mse_final:.4f}")
        print(f" MAE final : {calcular_mae(datos, w, b):.4f}")

    return w, b, historial

# ===========================================================================
# GRAFICOS
# ===========================================================================

def plot_modelo(datos, w, b, titulo="Regresión lineal"):
    import matplotlib.pyplot as plt

    # separar X e Y
    xs = [x for x, y in datos]
    ys = [y for x, y in datos]

    # puntos reales
    plt.figure()
    plt.scatter(xs, ys, label="Datos reales")

    # recta del modelo (ordenando X para que la línea no se cruce)
    xs_ordenados = sorted(xs)
    ys_pred = [predecir(x, w, b) for x in xs_ordenados]

    plt.plot(xs_ordenados, ys_pred, label=f"Modelo: y = {w:.2f}x + {b:.2f}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(titulo)
    plt.legend()
    plt.show()

def calcular_r2(datos, w, b):
    n = len(datos)
    y_media = sum(y for x, y in datos) / n
    ss_tot = sum((y - y_media)**2 for x, y in datos)
    ss_res = sum((y - predecir(x, w, b))**2 for x, y in datos)
    return 1 - (ss_res / ss_tot)