import math
import matplotlib.pyplot as plt

# ===========================================================================
# NORMALIZACION Z-SCORE
# =========================================================================== 

def calcular_media(valores):
    return sum(valores) / len(valores)

def calcular_sigma(valores, mu):
    return math.sqrt(sum((v - mu)**2 for v in valores) / len(valores))

def normalizar_zscore_multivariable(datos, num_features):

    # Separar columnas
    columnas_x = [[datos[i][j] for i in range(len(datos))] for j in range(num_features)]
    columna_y  = [datos[i][num_features] for i in range(len(datos))]  # y REAL

    params_x = []
    for col in columnas_x:
        mu    = calcular_media(col)
        sigma = calcular_sigma(col, mu)
        params_x.append((mu, sigma))

    datos_norm = []
    for fila in datos:
        xs_norm = tuple((fila[j] - params_x[j][0]) / params_x[j][1]
                        for j in range(num_features))
        y_real = fila[num_features]   # <-- y sin tocar
        datos_norm.append(xs_norm + (y_real,))

    return datos_norm, params_x

# ===========================================================================
# CONSTRUCCION MATRICIAL DEL DATASET
# ===========================================================================

def construir_matrices(datos_norm, num_features):

    X = [[1.0] + list(fila[:num_features]) for fila in datos_norm]
    y = [fila[num_features] for fila in datos_norm]
    return X, y

# ===========================================================================
# OPERACIONES MATRICIALES DESDE CERO
# ===========================================================================

def predecir_matricial(X, w):

    return [sum(X[i][j] * w[j] for j in range(len(w))) for i in range(len(X))]

def calcular_mse(y, y_hat):

    n = len(y)
    return sum((y[i] - y_hat[i])**2 for i in range(n)) / n

def calcular_gradiente(X, y, w):

    n = len(X)
    d = len(w)

    # e = y - X·w
    y_hat = predecir_matricial(X, w)
    e = [y[i] - y_hat[i] for i in range(n)]

    # ∇wJ = (-2/N) · X^T · e
    grad = [(-2.0 / n) * sum(X[i][j] * e[i] for i in range(n)) for j in range(d)]
    return grad

# ===========================================================================
# INICIALIZACION DE PESOS
# ===========================================================================

def init_historico(historico):
  
    return list(historico)

# ===========================================================================
# GRADIENT DESCENT MULTIVARIABLE
# ===========================================================================

def gradient_descent_multivariable(X, y, w_init, lr, iteraciones,
                                   epsilon=1e-8, verbose=True,
                                   mostrar_cada=300, nombre="GD"):

    w = list(w_init)
    d = len(w)
    historial = []   # (iteracion, lista_w, mse)

    if verbose:
        print("\n" + "=" * 75)
        print(f"GD MULTIVARIABLE — {nombre}")
        print("=" * 75)
        print(f" lr={lr} | N={len(X)} | d={d-1} features | "
              f"iter_max={iteraciones} | eps={epsilon}")
        etiquetas = ["w0(bias)", "w1(area_m2)", "w2(habitaciones)", "w3(antiguedad)", "w4(distancia_km)"]
        header = f" {'Iter':>6} " + " ".join(f"{etiquetas[j]:>11}" for j in range(d)) \
                 + f" {'MSE':>14} {'||∇w||':>12}"
        print(header)
        print(" " + "-"*6 + " " + " ".join(["-"*11]*d) + " " + "-"*14 + " " + "-"*12)

    iter_final = iteraciones

    for i in range(iteraciones):

        # --- Forward pass: y_hat = X · w    ---
        y_hat = predecir_matricial(X, w)

        # --- Costo: J = (1/N)||y - y_hat||²    ---
        mse = calcular_mse(y, y_hat)
        historial.append((i, list(w), mse))

        if verbose and (i % mostrar_cada == 0 or i < 5):
            w_str  = " ".join(f"{wj:>11.5f}" for wj in w)
            norm_g = math.sqrt(sum(g**2 for g in calcular_gradiente(X, y, w)))
            print(f" {i:>6} {w_str} {mse:>14.6f} {norm_g:>12.2e}")

        # --- Gradiente: ∇w = (-2/N) · X^T · e  ---
        grad = calcular_gradiente(X, y, w)

        # --- Criterio de parada: ||∇w|| < epsilon  ---
        norm_grad = math.sqrt(sum(g**2 for g in grad))
        if norm_grad < epsilon:
            if verbose:
                print(f"\n [Convergencia] iter={i} | ||∇w||={norm_grad:.2e} < eps={epsilon}")
            iter_final = i
            break

        # --- Actualizar: w <- w - α · ∇w ---
        w = [w[j] - lr * grad[j] for j in range(d)]

    # Estado final
    y_hat_f = predecir_matricial(X, w)
    mse_f   = calcular_mse(y, y_hat_f)
    historial.append((iter_final, list(w), mse_f))

    if verbose:
        w_str = " ".join(f"{wj:>11.5f}" for wj in w)
        print(f" {'FINAL':>6} {w_str} {mse_f:>14.6f}")
        print(f"\n Resultado {nombre}:")
        etiquetas = ["w0(bias)", "w1(area_m2)", "w2(habitaciones)", "w3(antiguedad)", "w4(distancia_km)"]
        for j, wj in enumerate(w):
            print(f"   {etiquetas[j]} = {wj:.6f}")
        print(f" MSE final: {mse_f:.6f}")

    return w, historial

# ===========================================================================
# GRAFICA DE CONVERGENCIA
# ===========================================================================

def graficar_convergencia_3_lr(X, y, w_hist):

    lr = 0.05  # único learning rate

    fig, ax = plt.subplots(figsize=(8, 6))

    _, hist = gradient_descent_multivariable(
        X, y, w_hist,
        lr=lr,
        iteraciones=2000,
        epsilon=1e-8,
        verbose=False
    )

    iters = [h[0] for h in hist]
    mses  = [h[2] for h in hist]

    ax.plot(iters, mses, linewidth=2, label="Histórico (GD)")

    ax.set_title(f"Convergencia GD — lr = {lr}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

# ===========================================================================
# R2
# ===========================================================================

def calcular_r2(y, y_hat):
    n = len(y)
    y_media = sum(y) / n
    ss_tot = sum((yi - y_media)**2 for yi in y)
    ss_res = sum((y[i] - y_hat[i])**2 for i in range(n))
    return 1 - (ss_res / ss_tot)