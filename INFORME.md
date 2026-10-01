# Informe del laboratorio de minería de datos

## Objetivo y alcance

Este trabajo implementa tres problemas de regresión lineal múltiple con los archivos `dolar_data.csv`, `glucosa_data.csv` y `energia_data.csv`. Para cada caso, el sistema separa los predictores `X` de la variable objetivo `y`, ajusta un modelo, calcula métricas sobre un conjunto reservado y permite consultar predicciones en una interfaz Streamlit.

## Desarrollo según CRISP-DM

1. **Comprensión del negocio:** estimar una variable numérica en tres contextos: precio del dólar, nivel de glucosa y consumo eléctrico.
2. **Comprensión de los datos:** se revisaron los encabezados especificados en la guía y el tipo numérico de las variables. Los CSV tienen 500 filas para dólar, 2.000 para glucosa y 10.000 para energía.
3. **Preparación:** el script conserva exclusivamente los predictores y el objetivo de cada ejercicio, convierte los datos a formato numérico, elimina filas incompletas y verifica que haya suficientes observaciones. Los nombres de columna se conservan tal como aparecen en los CSV.
4. **Modelado:** se ajusta `LinearRegression` de scikit-learn con 80 % de los datos. El 20 % restante se separa antes del ajuste para evaluar el modelo. La semilla `random_state=42` permite reproducir la partición.
5. **Evaluación:** se calculan MSE y R² para los tres ejercicios; además se calcula RMSE para expresar el error en las unidades originales del objetivo. El dashboard muestra métricas, ecuaciones, coeficientes y gráficas.
6. **Despliegue:** `joblib` serializa cada estimador con sus metadatos. Streamlit carga esos artefactos y permite ingresar valores y obtener una predicción.

## Resultados de evaluación

Los siguientes resultados corresponden a la ejecución de `python train_models.py` sobre los CSV incluidos. Las métricas se calcularon con el 20 % de prueba.

| Ejercicio | Ecuación estimada | MSE | RMSE | R² |
|---|---|---:|---:|---:|
| Dólar | `Precio_Dolar = 3985.783 + 4.984·Dia − 870.732·Inflacion − 1.377·Tasa_interes` | 2376.971 COP² | 48.754 COP | 0.9963 |
| Glucosa | `Nivel_Glucosa = 65.861 + 1.227·Edad + 0.933·IMC − 2.085·Actividad_Fisica` | 233.693 (mg/dL)² | 15.287 mg/dL | 0.6814 |
| Energía | `Consumo_Energia = 101.288 + 9.953·Temperatura + 5.020·Hora − 3.031·Dia_Semana` | 429.519 kWh² | 20.725 kWh | 0.8968 |

El intercepto `β₀` es la estimación cuando todos los predictores valen cero; ese escenario puede quedar fuera del rango observado y no siempre tiene interpretación práctica. Un coeficiente `βᵢ` expresa el cambio estimado de `y` por una unidad de `Xᵢ`, manteniendo constantes las demás variables. Por ejemplo, el coeficiente de inflación del dólar se expresa por una unidad completa de la tasa decimal; un punto porcentual equivale a 0,01 unidades.

Para comparar la magnitud relativa entre predictores que usan distintas unidades, se calcularon betas estandarizados con las desviaciones de cada CSV. El predictor con mayor magnitud estandarizada es `Dia` para dólar (aprox. 0,995), `Edad` para glucosa (aprox. 0,787) y `Temperatura` para energía (aprox. 0,780). Son comparaciones dentro de cada modelo y describen asociaciones lineales, no efectos causales.

## Interpretación y límites

- **Dólar:** el R² alto indica que el modelo explica gran parte de la variabilidad de la partición aleatoria de prueba. `Dia` tiene la asociación estandarizada más grande. Como la partición mezcla días de toda la serie, este resultado no comprueba por sí solo el desempeño al pronosticar fechas futuras; para ese uso conviene validar con una separación temporal.
- **Glucosa:** el modelo tiene ajuste moderado en esta partición. `Edad` presenta la asociación estandarizada más grande, seguida por `Actividad_Fisica` e `IMC`. El error típico en prueba es de unos 15,3 mg/dL. La predicción no es una medición ni un diagnóstico médico.
- **Energía:** el modelo explica una proporción alta de la variabilidad de prueba. `Temperatura` es el predictor con mayor asociación estandarizada. `Hora` está representada como número lineal aunque el ciclo diario vuelve de 24 a 1; por tanto, el modelo no representa explícitamente que esas horas son vecinas.

El MSE es el promedio de los errores elevados al cuadrado, por lo que sus unidades también quedan elevadas al cuadrado. El RMSE es la raíz del MSE y se expresa en las unidades de `y`. R² compara el modelo con la referencia de predecir la media; no es una probabilidad de acierto. Los resultados dependen de los datos, del modelo lineal y de la partición usada.

## Visualizaciones e interfaz

La aplicación presenta una dispersión de cada predictor contra su variable objetivo, como solicita la guía, además de una gráfica de valor real frente a predicción con una diagonal de referencia. También permite seleccionar los tres ejercicios, introducir los valores de `X` y calcular la estimación. Las tablas son HTML estático generado con `DataFrame.to_html(escape=True)` para evitar que las vistas tabulares dependan de PyArrow.

## Conclusiones

Los tres modelos se entrenan, serializan y consumen desde una interfaz única. El ajuste es muy alto para dólar, alto para energía y moderado para glucosa en la partición aleatoria utilizada. Los coeficientes y las gráficas ayudan a interpretar las asociaciones, con las limitaciones de causalidad, extrapolación, variables cíclicas y evaluación temporal señaladas arriba.

## Ejecución

```powershell
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```
