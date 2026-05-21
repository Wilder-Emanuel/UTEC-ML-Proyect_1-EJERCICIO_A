import csv
import Funciones_Univariable as fu
from Funciones_Multivariable import normalizar_zscore_multivariable
 
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
# MAIN
# ===========================================================================
 
datos = cargar_datos("Datos.csv")
 
datos_norm, _ = normalizar_zscore_multivariable(datos, 4)
 
variables = [
    ("Área (m²)",         [(fila[0], fila[4]) for fila in datos_norm]),
    ("Habitaciones",      [(fila[1], fila[4]) for fila in datos_norm]),
    ("Antigüedad (años)", [(fila[2], fila[4]) for fila in datos_norm]),
    ("Distancia (km)",    [(fila[3], fila[4]) for fila in datos_norm]),
]
 
for nombre, pares in variables:
    print(f"\n Analizando variable independiente: {nombre} vs Precio (USD)")
    w, b, _ = fu.gradient_descent(
        pares,
        lr          = 0.01,
        iteraciones = 1000,
        verbose     = True,
        mostrar_cada= 100,
    )
    r2 = fu.calcular_r2(pares, w, b)
    print(f" R² ({nombre}): {r2:.6f}")
    fu.plot_modelo(pares, w, b, titulo=f"Regresión Univariable — {nombre} vs Precio")

