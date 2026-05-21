import csv
import Funciones_Multivariable as fm

# ===========================================================================
# CARGA DEL DATASET
# ===========================================================================

def cargar_datos(ruta_csv):
    datos = []
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            area         = float(fila["Área (m²)"])
            habitaciones = float(fila["Habitaciones"])
            antiguedad   = float(fila["Antigüedad (años)"])
            distancia    = float(fila["Distancia (km)"])
            precio       = float(fila["Precio (USD)"])
            datos.append((area, habitaciones, antiguedad, distancia, precio))
    return datos

# ===========================================================================
# ECUACION NORMAL  w = (X^T X)^{-1} X^T y
# ===========================================================================

def transponer(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def multiplicar(A, B):
    filas_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(cols_A))
             for j in range(cols_B)] for i in range(filas_A)]

def invertir(M):
    n = len(M)
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(n):
            if r != col:
                f = aug[r][col]
                aug[r] = [aug[r][k] - f * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]

def ecuacion_normal(X, y):
    Xt    = transponer(X)
    XtX   = multiplicar(Xt, X)
    Xty   = multiplicar(Xt, [[yi] for yi in y])
    XtXi  = invertir(XtX)
    w_mat = multiplicar(XtXi, Xty)
    return [w_mat[j][0] for j in range(len(w_mat))]

# ===========================================================================
# MAIN
# ===========================================================================

NUM_FEATURES = 4
datos = cargar_datos("Datos.csv")

# -----------------------------------------------------------------------
# PUNTO 4 — Normalización Z-score + construcción de matrices
# -----------------------------------------------------------------------
print("\n" + "="*65)
print(" PUNTO 4 — PREPROCESAMIENTO: MATRIZ X (14×5) y VECTOR y")
print("="*65)

datos_norm, params_x = fm.normalizar_zscore_multivariable(datos, NUM_FEATURES)
X, y = fm.construir_matrices(datos_norm, NUM_FEATURES)

etiq_feat = ["Área (m²)", "Habitaciones", "Antigüedad (años)", "Distancia (km)"]
print("\n Parámetros Z-score por feature (μ, σ):")
for j, (mu, sigma) in enumerate(params_x):
    print(f"   {etiq_feat[j]:<22}  μ = {mu:10.4f}   σ = {sigma:10.4f}")

print(f"\n Dimensión X: {len(X)} × {len(X[0])}   |   Dimensión y: {len(y)} × 1")
print("\n Matriz X normalizada (primeras 5 filas):")
print(f" {'bias':>6} {'x1_area':>10} {'x2_hab':>10} {'x3_antig':>10} {'x4_dist':>10}")
for fila in X[:5]:
    print("  " + "  ".join(f"{v:>10.5f}" for v in fila))

# -----------------------------------------------------------------------
# PUNTO 5 — Ecuación Normal
# -----------------------------------------------------------------------
print("\n" + "="*65)
print(" PUNTO 5 — ECUACIÓN NORMAL: w = (X^T X)^{-1} X^T y")
print("="*65)

w_normal = ecuacion_normal(X, y)

etiq_w = ["w0 (bias)", "w1 (área)", "w2 (habitaciones)", "w3 (antigüedad)", "w4 (distancia)"]
print()
for j, wj in enumerate(w_normal):
    print(f"   {etiq_w[j]:<22} = {wj:>12.6f}")

y_hat_n = fm.predecir_matricial(X, w_normal)
mse_n   = fm.calcular_mse(y, y_hat_n)
print(f"\n MSE (Ecuación Normal): {mse_n:.6f}")

# -----------------------------------------------------------------------
# PUNTO 6 — Gradient Descent: w=0, lr=0.01, 5000 épocas
# -----------------------------------------------------------------------
print("\n" + "="*65)
print(" PUNTO 6 — GRADIENT DESCENT: w=0, lr=0.01, 5000 épocas")
print("="*65)

w_init = [227914.2853, 58015.8814, 13968.1997, 11910.0487, -12411.7082]

w_gd, _ = fm.gradient_descent_multivariable(
    X, y,
    w_init      = w_init,
    lr          = 0.01,
    iteraciones = 5000,
    epsilon     = 1e-8,
    verbose     = True,
    mostrar_cada= 500,
    nombre      = "Precios de Vivienda",
)

y_hat_gd = fm.predecir_matricial(X, w_gd)
r2_gd    = fm.calcular_r2(y, y_hat_gd)
print(f"\n R² (Gradient Descent): {r2_gd:.6f}")

# -----------------------------------------------------------------------
# PUNTO 7 — Interpretación Económica
# -----------------------------------------------------------------------
fm.graficar_convergencia_3_lr(X, y, w_init)

print("\n" + "="*65)
print(" PUNTO 7 — INTERPRETACIÓN ECONÓMICA DE LOS PESOS (GD)")
print("="*65)

etiq_eco = [
    (1, "w1 (área m²)",      "A mayor área, mayor precio.",               w_gd[1] > 0),
    (2, "w2 (habitaciones)", "Más habitaciones, mayor precio.",            w_gd[2] > 0),
    (3, "w3 (antigüedad)",   "Mayor antigüedad, menor precio esperado.",   w_gd[3] < 0),
    (4, "w4 (distancia km)", "Mayor distancia, menor precio (enunciado).", w_gd[4] < 0),
]
print()
for idx, etiq, explicacion, correcto in etiq_eco:
    signo  = f"{w_gd[idx]:+.6f}"
    estado = "✓ Coherente" if correcto else "✗ Revisar"
    print(f"   {etiq:<22}  w = {signo:>12}  →  {estado}")
    print(f"   {'':22}     {explicacion}")
    print()

# -----------------------------------------------------------------------
# PUNTO 8 — Predicción: 120 m², 3 hab, 10 años, 7 km
# -----------------------------------------------------------------------
print("\n" + "="*65)
print(" PUNTO 8 — PREDICCIÓN: 120 m², 3 hab, 10 años, 7 km")
print("="*65)

nueva_x    = [120.0, 3.0, 10.0, 7.0]
nueva_norm = [(nueva_x[j] - params_x[j][0]) / params_x[j][1] for j in range(NUM_FEATURES)]
fila_nueva = [1.0] + nueva_norm

precio_pred = sum(w_gd[j] * fila_nueva[j] for j in range(len(w_gd)))

print(f"\n   Área        : {nueva_x[0]:.1f} m²")
print(f"   Habitaciones: {nueva_x[1]:.0f}")
print(f"   Antigüedad  : {nueva_x[2]:.1f} años")
print(f"   Distancia   : {nueva_x[3]:.1f} km")
print(f"\n   Precio estimado: ${precio_pred:,.2f} USD")

