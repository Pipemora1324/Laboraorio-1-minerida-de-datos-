"""Dashboard educativo de predicción para los modelos del laboratorio.

Inicie desde esta carpeta con: streamlit run app.py
El dashboard muestra preparación, entradas, predicciones y evaluación CRISP-DM.
"""

# Streamlit construye la interfaz web interactiva; pandas organiza los datos;
# joblib carga los modelos persistidos; Plotly dibuja dispersiones interactivas.
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


CARPETA_PROYECTO = Path(__file__).resolve().parent

# El mismo esquema que train_models.py garantiza que la selección de módulo
# encuentre el archivo y las columnas exactas que espera cada regresión.
MODULOS = {
    "💵 Dólar Predictor": {
        "csv": "dolar_data.csv", "modelo": "modelo_dolar.joblib",
        "variables": ["Dia", "Inflacion", "Tasa_interes"],
        "objetivo": "Precio_Dolar", "unidad": "COP", "decimales": 2,
    },
    "🩸 Glucosa Predictor": {
        "csv": "glucosa_data.csv", "modelo": "modelo_glucosa.joblib",
        "variables": ["Edad", "IMC", "Actividad_Fisica"],
        "objetivo": "Nivel_Glucosa", "unidad": "mg/dL", "decimales": 2,
    },
    "⚡ Energía Predictor": {
        "csv": "energia_data.csv", "modelo": "modelo_energia.joblib",
        "variables": ["Temperatura", "Hora", "Dia_Semana"],
        "objetivo": "Consumo_Energia", "unidad": "kWh", "decimales": 2,
    },
}


def aplicar_estilos() -> None:
    """Inyecta el tema oscuro y los acentos del diseño solicitado."""
    # CSS acota los estilos a elementos visuales de Streamlit y establece una
    # jerarquía legible con fondos #0B0F17, #111827 y #1F2937.
    st.markdown("""
    <style>
    .stApp { background: #0B0F17; color: #E5E7EB; }
    [data-testid="stHeader"] { background: rgba(11,15,23,.9); }
    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1F2937; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #1F2937; border-radius: 16px;
            background: linear-gradient(120deg,#111827,#0f2028); margin-bottom: 1rem; }
    .eyebrow { color: #06B6D4; text-transform: uppercase; letter-spacing: .13em;
               font-size: .75rem; font-weight: 700; }
    .hero h1 { margin: .35rem 0; color: #F9FAFB; font-size: 1.8rem; }
    .hero p { margin: 0; color: #9CA3AF; }
    .status { display: inline-block; color: #10B981; background: #052e26;
              border: 1px solid #065f46; border-radius: 999px; padding: .32rem .7rem;
              font-size: .82rem; font-weight: 700; }
    div.stButton > button { background: #10B981; color: #04130f; border: 0;
                            font-weight: 700; border-radius: 9px; }
    div.stButton > button:hover { background: #34D399; color: #04130f; }
    [data-testid="stMetric"] { background: #111827; border: 1px solid #1F2937;
                                padding: .9rem; border-radius: 12px; }
    .tabla-laboratorio { width: 100%; border-collapse: collapse; color: #E5E7EB;
                         background: #111827; border: 1px solid #1F2937;
                         border-radius: 10px; overflow: hidden; font-size: .9rem; }
    .tabla-laboratorio th { background: #1F2937; color: #F9FAFB; text-align: left;
                            padding: .7rem .8rem; }
    .tabla-laboratorio td { padding: .6rem .8rem; border-top: 1px solid #1F2937; }
    .tabla-laboratorio tr:hover td { background: #172235; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo(ruta: str):
    """Carga una sola vez el artefacto en memoria para acelerar interacciones."""
    # joblib restaura el estimador y los metadatos creados por train_models.py.
    return joblib.load(ruta)


@st.cache_data
def cargar_csv(ruta: str) -> pd.DataFrame:
    """Lee el CSV para mostrar una vista previa y construir la gráfica."""
    return pd.read_csv(ruta)


def mostrar_tabla_estatica(datos: pd.DataFrame) -> None:
    """Renderiza un DataFrame como HTML seguro, sin el serializador Arrow.

    Tanto st.dataframe como st.table pueden delegar la conversión tabular a
    PyArrow según la versión instalada de Streamlit. Esta vista es estática,
    por eso generamos HTML desde pandas y lo enviamos como marcado; escape=True
    convierte cualquier texto del CSV en texto literal para evitar inyectar HTML.
    """
    html_tabla = datos.to_html(
        index=False,
        border=0,
        classes="tabla-laboratorio",
        escape=True,
        justify="left",
    )
    st.markdown(html_tabla, unsafe_allow_html=True)


def crear_entrada(nombre: str, datos: pd.DataFrame, valores: dict) -> None:
    """Dibuja un control apropiado para cada predictor y guarda su valor."""
    # Rangos obtenidos de los datos observados hacen que los controles sean
    # útiles para explorar el dataset; si no hay datos se usan límites seguros.
    rangos = {
        "Dia": (1, 366), "Inflacion": (-0.10, 0.20), "Tasa_interes": (0.0, 30.0),
        "Edad": (1, 110), "IMC": (10.0, 60.0), "Actividad_Fisica": (0.0, 24.0),
        "Temperatura": (-20.0, 60.0), "Hora": (0, 23), "Dia_Semana": (1, 7),
    }
    es_entero = nombre in {"Dia", "Edad", "Hora", "Dia_Semana"}
    minimo_def, maximo_def = rangos[nombre]
    if nombre in datos.columns:
        minimo, maximo = float(datos[nombre].min()), float(datos[nombre].max())
        # Se amplía un rango constante para que el widget siga teniendo amplitud.
        if minimo == maximo:
            minimo, maximo = minimo - 1, maximo + 1
    else:
        minimo, maximo = minimo_def, maximo_def
    # El valor inicial es la mediana empírica, un punto representativo y robusto
    # frente a valores extremos; si falta el CSV se usa el centro del rango.
    inicial = float(datos[nombre].median()) if nombre in datos.columns else (minimo + maximo) / 2
    # Streamlit crea la clave de estado asociada al widget cuando se le pasa
    # key=nombre. Aun así, inicializamos explícitamente el estado antes de
    # dibujarlo para que el valor inicial sea claro y nunca se lea una clave
    # inexistente durante la primera carga de la aplicación.
    if es_entero:
        minimo_widget, maximo_widget = int(minimo), int(maximo)
        valor_inicial = max(minimo_widget, min(int(round(inicial)), maximo_widget))
    else:
        minimo_widget, maximo_widget = float(minimo), float(maximo)
        valor_inicial = max(minimo_widget, min(inicial, maximo_widget))
    st.session_state.setdefault(nombre, valor_inicial)

    # Con key= el widget y session_state comparten el valor. No se pasa además
    # value=, porque eso crearía dos fuentes de verdad para la entrada.
    if es_entero:
        valor = st.number_input(
            nombre, min_value=minimo_widget, max_value=maximo_widget,
            step=1, key=nombre,
        )
    else:
        valor = st.number_input(
            nombre, min_value=minimo_widget, max_value=maximo_widget,
            step=0.1, format="%.3f", key=nombre,
        )
    # Se usa el valor retornado por el widget; así la función no depende de
    # releer una clave de estado antes de que Streamlit la haya registrado.
    valores[nombre] = valor


def dibujar_resultado(
    artefacto: dict,
    configuracion: dict,
    prediccion: float,
    datos: pd.DataFrame,
) -> None:
    """Presenta la predicción, calidad, coeficientes y dispersión de prueba."""
    # La cifra principal es la salida del modelo para los valores ingresados.
    st.subheader("Resultado de la predicción")
    st.metric(configuracion["objetivo"], f"{prediccion:,.{configuracion['decimales']}f} {configuracion['unidad']}")

    # Las métricas provienen del conjunto de prueba, no de la entrada actual.
    # Esto deja claro que resumen el desempeño global y no la certeza individual.
    metricas = artefacto["metricas"]
    columnas = st.columns(3)
    columnas[0].metric("R² · prueba", f"{metricas['R2']:.4f}", help="Proporción de variabilidad explicada en los datos de prueba.")
    columnas[1].metric("MSE · prueba", f"{metricas['MSE']:.4f}", help="Promedio del error al cuadrado; queda en unidades al cuadrado.")
    columnas[2].metric("RMSE · prueba", f"{metricas['RMSE']:.4f}", help="Raíz del MSE; se expresa en las mismas unidades del objetivo.")

    st.markdown("#### Interpretación del modelo")
    st.caption(f"Intercepto β₀ = {artefacto['intercepto']:.5g}: estimación cuando todos los predictores valen cero. La utilidad de ese caso depende del contexto.")
    # Los coeficientes en unidades originales no se comparan directamente si
    # las variables usan escalas distintas. El beta estandarizado multiplica
    # cada coeficiente por la desviación de X y divide por la de y; su magnitud
    # facilita ordenar asociaciones lineales en la escala de desviaciones.
    desviacion_objetivo = float(datos[configuracion["objetivo"]].std(ddof=0))
    filas_coeficientes = []
    for nombre, coef in artefacto["coeficientes"].items():
        desviacion_variable = float(datos[nombre].std(ddof=0))
        beta_estandarizado = (
            float(coef * desviacion_variable / desviacion_objetivo)
            if desviacion_objetivo > 0
            else 0.0
        )
        filas_coeficientes.append({
            "Variable": nombre,
            "Coeficiente βᵢ": float(coef),
            "Beta estandarizado": beta_estandarizado,
            "Interpretación": f"Por cada unidad adicional de {nombre}, la predicción cambia {coef:.5g}, manteniendo constantes las demás variables.",
        })
    coeficientes = pd.DataFrame(filas_coeficientes)
    # La tabla de coeficientes también pasa por HTML estático: así ninguna de
    # las vistas tabulares pide a Streamlit serializar un DataFrame con Arrow.
    mostrar_tabla_estatica(coeficientes)
    variable_mayor_impacto = max(
        artefacto["coeficientes"],
        key=lambda nombre: abs(
            artefacto["coeficientes"][nombre]
            * float(datos[nombre].std(ddof=0))
            / desviacion_objetivo
        ) if desviacion_objetivo > 0 else 0.0,
    )
    st.caption(
        f"Mayor asociación lineal estandarizada en este dataset: {variable_mayor_impacto}. "
        "Los coeficientes describen asociaciones aprendidas; no demuestran causalidad."
    )

    # El PDF solicita graficar cada variable independiente frente al objetivo.
    # Cada dispersión usa todo el CSV y su propio eje X para respetar las
    # unidades particulares (edad, temperatura, inflación, etc.) de cada campo.
    st.markdown("#### Relación de cada predictor con el objetivo")
    columnas_graficas = st.columns(len(configuracion["variables"]))
    for columna, variable in zip(columnas_graficas, configuracion["variables"]):
        figura_variable = px.scatter(
            datos,
            x=variable,
            y=configuracion["objetivo"],
            opacity=0.55,
            color_discrete_sequence=["#06B6D4"],
            template="plotly_dark",
            title=f"{variable} vs. {configuracion['objetivo']}",
        )
        figura_variable.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            margin={"l": 10, "r": 10, "t": 48, "b": 10},
            height=300,
        )
        columna.plotly_chart(figura_variable, width="stretch")

    # La dispersión compara las respuestas observadas y estimadas en prueba.
    # La diagonal ideal sirve de referencia: puntos próximos indican menor error.
    st.markdown("#### Predicción frente a valor real · conjunto de prueba")
    grafica = pd.DataFrame({"Valor real": artefacto["y_real_prueba"], "Predicción": artefacto["y_predicha_prueba"]})
    figura = px.scatter(grafica, x="Valor real", y="Predicción", opacity=0.75,
                        color_discrete_sequence=["#06B6D4"], template="plotly_dark")
    minimo = min(grafica.min())
    maximo = max(grafica.max())
    figura.add_shape(type="line", x0=minimo, y0=minimo, x1=maximo, y1=maximo,
                     line={"color": "#10B981", "dash": "dash"})
    figura.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", margin={"l": 10, "r": 10, "t": 20, "b": 10})
    st.plotly_chart(figura, width="stretch")


def main() -> None:
    """Orquesta navegación, carga, controles y predicción dentro de Streamlit."""
    st.set_page_config(page_title="Data Mining Lab | CRISP-DM", page_icon="📊", layout="wide")
    aplicar_estilos()

    # La cabecera da contexto al laboratorio y el indicador refleja si existen
    # los tres artefactos esperados para usar la aplicación completa.
    todos_listos = all((CARPETA_PROYECTO / item["modelo"]).exists() for item in MODULOS.values())
    etiqueta_estado = "● MODELOS ACTIVOS" if todos_listos else "● ENTRENAMIENTO PENDIENTE"
    st.markdown(f"""
    <div class="hero"><span class="eyebrow">Laboratorio de minería de datos · CRISP-DM</span>
    <h1>Data Mining Lab | CRISP-DM Linear Regression Models</h1>
    <p>Explora datos, estima resultados e interpreta modelos de regresión lineal múltiple.</p>
    <br><span class="status">{etiqueta_estado}</span></div>
    """, unsafe_allow_html=True)

    # El selector conmuta los tres problemas predictivos sin mezclar variables.
    nombre_modulo = st.selectbox("Selecciona un módulo", list(MODULOS.keys()))
    configuracion = MODULOS[nombre_modulo]
    ruta_modelo = CARPETA_PROYECTO / configuracion["modelo"]
    ruta_csv = CARPETA_PROYECTO / configuracion["csv"]

    if not ruta_modelo.exists():
        st.warning(f"No existe {configuracion['modelo']}. Ejecuta primero `python train_models.py`.")
        st.stop()
    try:
        artefacto = cargar_modelo(str(ruta_modelo))
        datos = cargar_csv(str(ruta_csv))
    except Exception as error:
        st.error(f"No se pudo cargar el modelo o el CSV: {error}")
        st.stop()

    # Las entradas viven en la columna lateral; se crea una fila en el formato
    # y el orden exacto usado durante el entrenamiento antes de llamar predict.
    st.sidebar.header("Parámetros de entrada")
    valores = {}
    for variable in configuracion["variables"]:
        # Se pasan por separado los datos de referencia y el diccionario que
        # recoge las entradas para iniciar cada control con la mediana real.
        crear_entrada(variable, datos, valores)
    st.sidebar.caption("Los valores iniciales corresponden a la mediana del CSV.")
    calcular = st.sidebar.button("Calcular Predicción", width="stretch")

    izquierda, derecha = st.columns([1.15, 1])
    with izquierda:
        st.subheader("Datos del módulo")
        st.caption(f"{len(datos):,} filas · {len(configuracion['variables'])} variables predictoras")
        # Se muestran las primeras ocho observaciones como HTML estático para
        # no activar la conversión interna de DataFrame que requiere PyArrow.
        mostrar_tabla_estatica(datos.head(8))
        st.markdown("**Ecuación aprendida**")
        st.code(artefacto["formula"], language="text")
    with derecha:
        st.subheader("Variables del modelo")
        st.write(f"**X (predictoras):** {', '.join(configuracion['variables'])}")
        st.write(f"**y (objetivo):** {configuracion['objetivo']}")
        st.info("X aporta las características de entrada; y es la cantidad que el modelo intenta estimar.")

    if calcular:
        # DataFrame de una fila y nombres exactos preservan el orden/esquema con
        # el que se ajustó LinearRegression en el script de entrenamiento.
        entrada = pd.DataFrame([[valores[col] for col in configuracion["variables"]]],
                               columns=configuracion["variables"])
        prediccion = float(artefacto["modelo"].predict(entrada)[0])
        st.session_state[f"prediccion_{nombre_modulo}"] = prediccion

    clave_resultado = f"prediccion_{nombre_modulo}"
    if clave_resultado in st.session_state:
        dibujar_resultado(
            artefacto,
            configuracion,
            st.session_state[clave_resultado],
            datos,
        )
    elif nombre_modulo == "🩸 Glucosa Predictor":
        st.caption("La alerta orientativa de rango aparecerá después de calcular; no sustituye una evaluación clínica.")

    # Umbrales explícitamente orientativos: la glucosa depende de condiciones de
    # medición (ayuno/posprandial). Por eso no se etiqueta como diagnóstico.
    if nombre_modulo == "🩸 Glucosa Predictor" and clave_resultado in st.session_state:
        valor = st.session_state[clave_resultado]
        if valor < 70:
            st.warning("Predicción por debajo de 70 mg/dL: rango bajo orientativo. Confirma el contexto de medición con un profesional.")
        elif valor >= 126:
            st.warning("Predicción de 126 mg/dL o más: puede ser elevada si la medición fue en ayunas. Una predicción no permite diagnosticar.")
        else:
            st.success("Predicción entre 70 y 125 mg/dL. La interpretación depende del momento de medición y del contexto clínico.")

    # En el pie se explican brevemente las medidas para apoyar la lectura crítica.
    with st.expander("¿Cómo leer las métricas?"):
        st.markdown("- **MSE:** promedio de errores cuadrados; menor suele indicar mejor ajuste, aunque sus unidades están elevadas al cuadrado.\n- **RMSE:** error típico expresado en las unidades de la variable objetivo.\n- **R²:** fracción de variabilidad explicada en prueba; puede ser negativa si el modelo rinde peor que predecir la media.\n- Las métricas son una evaluación general del conjunto de prueba, no una probabilidad de acierto para una fila individual.")


# Streamlit importa y ejecuta el archivo como aplicación; este guard también
# permite inspeccionar las funciones sin lanzar el servidor desde Python.
if __name__ == "__main__":
    main()
