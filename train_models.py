"""Entrenamiento reproducible de los tres modelos del laboratorio CRISP-DM.

Flujo CRISP-DM cubierto aquí: comprensión/preparación de datos, modelado,
evaluación y despliegue mediante archivos serializados que consume app.py.
Ejecute desde esta carpeta con: python train_models.py
"""

# pandas permite leer y revisar los datos tabulares; scikit-learn proporciona
# la partición, la regresión y las métricas; joblib guarda objetos Python
# entrenados para reutilizarlos sin volver a ajustar el modelo en el dashboard.
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Ruta basada en la ubicación del script: así funciona incluso si el usuario
# lo ejecuta desde otra carpeta y evita depender del directorio actual.
CARPETA_PROYECTO = Path(__file__).resolve().parent

# Para cada caso se declaran los nombres exactos de archivo, predictores (X),
# objetivo (y), artefacto de salida y una etiqueta comprensible.
CONFIGURACIONES = {
    "dolar": {
        "archivo": "dolar_data.csv",
        "variables": ["Dia", "Inflacion", "Tasa_interes"],
        "objetivo": "Precio_Dolar",
        "modelo": "modelo_dolar.joblib",
        "etiqueta": "Precio del dólar",
    },
    "glucosa": {
        "archivo": "glucosa_data.csv",
        "variables": ["Edad", "IMC", "Actividad_Fisica"],
        "objetivo": "Nivel_Glucosa",
        "modelo": "modelo_glucosa.joblib",
        "etiqueta": "Nivel de glucosa",
    },
    "energia": {
        "archivo": "energia_data.csv",
        "variables": ["Temperatura", "Hora", "Dia_Semana"],
        "objetivo": "Consumo_Energia",
        "modelo": "modelo_energia.joblib",
        "etiqueta": "Consumo de energía",
    },
}


def cargar_datos(configuracion: dict) -> pd.DataFrame:
    """Lee un CSV y valida columnas y valores antes de iniciar el modelado."""
    ruta = CARPETA_PROYECTO / configuracion["archivo"]
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {ruta}")

    # La primera fila contiene los encabezados. Se seleccionan las columnas
    # declaradas para detectar errores de escritura y mantener el esquema.
    datos = pd.read_csv(ruta)
    columnas_requeridas = configuracion["variables"] + [configuracion["objetivo"]]
    faltantes = [col for col in columnas_requeridas if col not in datos.columns]
    if faltantes:
        raise ValueError(f"En {ruta.name} faltan columnas: {faltantes}")

    # La regresión necesita números finitos. Convertimos errores de lectura en
    # nulos, descartamos filas incompletas y rechazamos infinitos explícitamente.
    datos = datos[columnas_requeridas].apply(pd.to_numeric, errors="coerce")
    datos = datos.dropna()
    if datos.empty:
        raise ValueError(f"{ruta.name} no contiene filas numéricas completas.")
    if not datos.map(lambda valor: pd.notna(valor) and abs(valor) != float("inf")).all().all():
        raise ValueError(f"{ruta.name} contiene valores infinitos.")
    if len(datos) < 5:
        raise ValueError(f"{ruta.name} necesita al menos 5 filas para entrenar y evaluar.")
    return datos


def entrenar_modelo(nombre: str, configuracion: dict) -> dict:
    """Separa X/y, ajusta una regresión, mide su desempeño y serializa todo."""
    datos = cargar_datos(configuracion)

    # X contiene las columnas que el modelo observa para explicar el fenómeno;
    # y contiene la respuesta que queremos estimar. Mantenerlas separadas evita
    # que el modelo reciba accidentalmente la respuesta como entrada.
    X = datos[configuracion["variables"]]
    y = datos[configuracion["objetivo"]]

    # Reservar una parte de los datos para evaluación estima el comportamiento
    # sobre observaciones no usadas en el ajuste. random_state hace reproducible
    # la comparación; test_size asigna 20 % a prueba y 80 % a entrenamiento.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Regresión lineal múltiple estima una pendiente por predictor y un término
    # constante minimizando la suma de errores cuadrados en entrenamiento.
    modelo = LinearRegression()
    modelo.fit(X_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(X_prueba)

    # MSE es el promedio de errores al cuadrado (penaliza más los errores grandes);
    # RMSE es su raíz y queda en las unidades de y; R² compara el ajuste con usar
    # la media como referencia (1 es perfecto, 0 equivale a esa referencia).
    mse = float(mean_squared_error(y_prueba, predicciones))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_prueba, predicciones))

    # En y = β0 + β1*x1 + ... + βp*xp, β0 es la predicción cuando todos los
    # predictores valen cero (su sentido práctico depende de si ese caso existe).
    # Cada βi indica cuánto cambia la predicción por una unidad adicional de xi,
    # manteniendo constantes los demás predictores; asociación no implica causa.
    terminos = [f"({coef:.6g} × {variable})" for variable, coef in zip(X.columns, modelo.coef_)]
    formula = f"{configuracion['objetivo']} = {modelo.intercept_:.6g} + " + " + ".join(terminos)

    # Se guarda el estimador junto con el esquema, las métricas y la muestra de
    # prueba; el dashboard obtiene de un solo artefacto todo lo necesario.
    artefacto = {
        "modelo": modelo,
        "variables": list(X.columns),
        "objetivo": configuracion["objetivo"],
        "etiqueta": configuracion["etiqueta"],
        "metricas": {"MSE": mse, "RMSE": rmse, "R2": r2},
        "intercepto": float(modelo.intercept_),
        "coeficientes": {col: float(coef) for col, coef in zip(X.columns, modelo.coef_)},
        "formula": formula,
        "y_real_prueba": y_prueba.tolist(),
        "y_predicha_prueba": predicciones.tolist(),
        "filas_utilizadas": int(len(datos)),
    }
    ruta_salida = CARPETA_PROYECTO / configuracion["modelo"]
    joblib.dump(artefacto, ruta_salida)

    # La salida de consola hace visible la ecuación aprendida y las medidas
    # para documentar el resultado de cada ciclo de entrenamiento.
    print(f"\n{'=' * 72}\nModelo: {nombre.title()} | filas válidas: {len(datos)}")
    print(f"Fórmula: {formula}")
    # Se usa texto ASCII en las etiquetas de consola para funcionar también
    # en terminales Windows configuradas con páginas de códigos antiguas.
    print(f"Intercepto (beta 0): {modelo.intercept_:.6f}")
    for variable, coeficiente in artefacto["coeficientes"].items():
        print(f"Coeficiente beta para {variable}: {coeficiente:.6f}")
    print(f"MSE: {mse:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}")
    print(f"Artefacto guardado en: {ruta_salida}")
    return artefacto


def main() -> None:
    """Ejecuta una vez cada configuración y reporta fallos con claridad."""
    print("Entrenamiento de modelos de regresión lineal múltiple (CRISP-DM)")
    for nombre, configuracion in CONFIGURACIONES.items():
        entrenar_modelo(nombre, configuracion)
    print("\nEntrenamiento finalizado. Los tres archivos .joblib están listos.")


# Este guard permite importar funciones desde otro módulo sin iniciar el ajuste.
if __name__ == "__main__":
    main()
