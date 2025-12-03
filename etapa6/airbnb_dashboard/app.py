from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------------------------------
# 1. Cargar datos y modelos
# ----------------------------------------------------

# Dataset reducido para gráficos
df = pd.read_csv("data/listings_dashboard_sample.csv")

# Modelos y columnas de Etapa 4 (los .pkl que guardaste)
modelo_regresion = joblib.load("models/modelo_regresion.pkl")
columnas_regresion = joblib.load("models/columnas_regresion.pkl")

modelo_clasificacion = joblib.load("models/modelo_clasificacion.pkl")
columnas_clasificacion = joblib.load("models/columnas_clasificacion.pkl")

# Listas para los controles
neighbourhoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
room_types = sorted(df["room_type"].dropna().unique())

acc_min = int(df["accommodates"].min())
acc_max = int(df["accommodates"].max())


# ----------------------------------------------------
# 2. Crear app Dash
# ----------------------------------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"padding": "20px", "backgroundColor": "#f5f9ff"},
    children=[
        html.Div(
            style={
                "backgroundColor": "#d7e9ff",
                "padding": "20px",
                "borderRadius": "8px",
                "marginBottom": "20px",
            },
            children=[
                html.H1(
                    "Tablero de Analítica Airbnb – Ciudad de México",
                    style={"textAlign": "center", "color": "#111827"},
                ),
                html.P(
                    "Tablero para anfitriones de Airbnb que desean estimar el precio sugerido "
                    "por noche y saber si su anuncio es competitivo en su zona.",
                    style={"textAlign": "center"},
                ),
            ],
        ),

        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                # -----------------------------------------
                # COLUMNA IZQUIERDA: Inputs del anuncio
                # -----------------------------------------
                html.Div(
                    style={
                        "flex": "1",
                        "backgroundColor": "white",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "border": "1px solid #e5e7eb",
                    },
                    children=[
                        html.H2("Datos del anuncio"),

                        html.Label("Barrio"),
                        dcc.Dropdown(
                            id="dd-barrio",
                            options=[{"label": b, "value": b} for b in neighbourhoods],
                            value=neighbourhoods[0],
                            style={"marginBottom": "10px"},
                        ),

                        html.Label("Tipo de habitación"),
                        dcc.Dropdown(
                            id="dd-room",
                            options=[{"label": rt, "value": rt} for rt in room_types],
                            value=room_types[0],
                            style={"marginBottom": "10px"},
                        ),

                        html.Label("Capacidad (accommodates)"),
                        dcc.Slider(
                            id="slider-acc",
                            min=acc_min,
                            max=acc_max,
                            value=2,
                            step=1,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Mínimo de noches"),
                        dcc.Input(
                            id="input-min-nights",
                            type="number",
                            min=1,
                            step=1,
                            value=1,
                            style={"width": "100%", "marginBottom": "10px"},
                        ),

                        html.Label("Disponibilidad anual (días al año)"),
                        dcc.Slider(
                            id="slider-avail",
                            min=0,
                            max=365,
                            value=180,
                            step=10,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("¿Eres Superhost?"),
                        dcc.Checklist(
                            id="check-superhost",
                            options=[{"label": "Soy superhost", "value": "yes"}],
                            value=[],
                            style={"marginBottom": "10px"},
                        ),

                        html.Label("Tasa de respuesta del host (%)"),
                        dcc.Slider(
                            id="slider-response",
                            min=0,
                            max=100,
                            step=5,
                            value=90,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Tasa de aceptación del host (%)"),
                        dcc.Slider(
                            id="slider-accept",
                            min=0,
                            max=100,
                            step=5,
                            value=95,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Button(
                            "Calcular",
                            id="btn-calcular",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "backgroundColor": "#1f2937",
                                "color": "white",
                                "borderRadius": "6px",
                                "border": "none",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),

                # -----------------------------------------
                # COLUMNA DERECHA: Resultados + Gráficas
                # -----------------------------------------
                html.Div(
                    style={"flex": "2", "display": "flex", "flexDirection": "column", "gap": "20px"},
                    children=[
                        # Resultados
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "8px",
                                "border": "1px solid #e5e7eb",
                            },
                            children=[
                                html.H2("Resultados de modelos"),
                                html.Div(
                                    style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={
                                                "flex": "1",
                                                "border": "1px solid #e5e7eb",
                                                "borderRadius": "8px",
                                                "padding": "15px",
                                            },
                                            children=[
                                                html.H4("Precio sugerido por noche"),
                                                html.Div(
                                                    id="out-precio",
                                                    style={
                                                        "fontSize": "24px",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "10px",
                                                    },
                                                ),
                                                html.Div(
                                                    id="out-precio-texto",
                                                    style={"color": "#4b5563"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={
                                                "flex": "1",
                                                "border": "1px solid #e5e7eb",
                                                "borderRadius": "8px",
                                                "padding": "15px",
                                            },
                                            children=[
                                                html.H4("Recomendación"),
                                                html.Div(
                                                    id="out-rec",
                                                    style={
                                                        "fontSize": "20px",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "10px",
                                                    },
                                                ),
                                                html.Div(
                                                    id="out-rec-texto",
                                                    style={"color": "#4b5563"},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # Visualizaciones
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "8px",
                                "border": "1px solid #e5e7eb",
                            },
                            children=[
                                html.H2("Visualizaciones"),
                                html.Div(
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "20px"},
                                    children=[
                                        html.Div(
                                            style={"minWidth": "0"},
                                            children=[
                                                html.H4("Precio por tipo de habitación"),
                                                dcc.Graph(id="graf-roomtype", style={"height": "400px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={"minWidth": "0"},
                                            children=[
                                                html.H4("Precio por huésped por barrio"),
                                                dcc.Graph(id="graf-barrio", style={"height": "400px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={"minWidth": "0"},
                                            children=[
                                                html.H4("Capacidad vs precio"),
                                                dcc.Graph(id="graf-acc-price", style={"height": "400px"}),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ----------------------------------------------------
# 3. Función auxiliar: construir fila y dummies
# ----------------------------------------------------
def construir_fila_features(
    barrio,
    room,
    accommodates,
    min_nights,
    avail,
    is_superhost,
    resp_pct,
    acc_pct,
):
    """
    Crea un DataFrame de UNA fila con las columnas crudas
    y luego hace get_dummies igual que en Etapa 4 para alinear
    con las columnas guardadas en columnas_regresion / columnas_clasificacion.
    """
    # host_is_superhost a 0/1
    host_is_superhost = 1 if is_superhost else 0

    # Tasas de respuesta/aceptación en proporción 0–1
    host_response_rate = (resp_pct or 0) / 100
    host_acceptance_rate = (acc_pct or 0) / 100

    base = pd.DataFrame(
        [
            {
                "accommodates": accommodates,
                "minimum_nights": min_nights,
                "availability_365": avail,
                # Si usaste estas variables en el modelo original y las tienes en el CSV,
                # podrías agregarlas aquí. Las dejo como 0 si no las estás usando:
                "estimated_occupancy_l365d": 0,
                "estimated_revenue_l365d": 0,
                "host_response_rate": host_response_rate,
                "host_acceptance_rate": host_acceptance_rate,
                "host_is_superhost": host_is_superhost,
                "room_type": room,
                "neighbourhood_cleansed": barrio,
            }
        ]
    )

    # get_dummies igual que en tu notebook (sin drop_first aquí;
    # las columnas exactas las alineamos usando las listas guardadas)
    base_dummies = pd.get_dummies(
        base,
        columns=["room_type", "neighbourhood_cleansed"],
        drop_first=False,
    )

    return base_dummies


# ----------------------------------------------------
# 4. Callbacks – Predicciones
# ----------------------------------------------------
@app.callback(
    Output("out-precio", "children"),
    Output("out-precio-texto", "children"),
    Output("out-rec", "children"),
    Output("out-rec-texto", "children"),
    Input("btn-calcular", "n_clicks"),
    State("dd-barrio", "value"),
    State("dd-room", "value"),
    State("slider-acc", "value"),
    State("input-min-nights", "value"),
    State("slider-avail", "value"),
    State("check-superhost", "value"),
    State("slider-response", "value"),
    State("slider-accept", "value"),
)
def actualizar_predicciones(
    n_clicks,
    barrio,
    room,
    accommodates,
    min_nights,
    avail,
    superhost_values,
    resp_pct,
    acc_pct,
):
    if not n_clicks:
        return "", "Introduce los datos del anuncio y presiona Calcular.", "", ""

    is_superhost = "yes" in (superhost_values or [])

    # 1) Construir fila con mismas transformaciones de dummies
    fila = construir_fila_features(
        barrio=barrio,
        room=room,
        accommodates=accommodates,
        min_nights=min_nights,
        avail=avail,
        is_superhost=is_superhost,
        resp_pct=resp_pct,
        acc_pct=acc_pct,
    )

    # 2) Alinear columnas con el modelo de regresión
    X_reg = fila.reindex(columns=columnas_regresion, fill_value=0)
    # Predicción: asumo que tu modelo usa log(price)
    log_price = modelo_regresion.predict(X_reg)[0]
    price = float(np.exp(log_price))  # si tu modelo ya predice price directo, usa: price = float(log_price)

    # 3) Alinear columnas con el modelo de clasificación
    X_clf = fila.reindex(columns=columnas_clasificacion, fill_value=0)
    proba_recom = modelo_clasificacion.predict_proba(X_clf)[0, 1]
    es_recomendable = proba_recom >= 0.5

    # 4) Armar textos bonitos
    texto_precio = f"${price:,.0f} MXN / noche"
    texto_precio_detalle = (
        "Este es el precio sugerido según propiedades similares con características "
        "parecidas en tu zona (modelo de regresión en log(price))."
    )

    if es_recomendable:
        texto_rec = "✔ Recomendada"
        texto_rec_detalle = (
            f"La probabilidad estimada de que tu propiedad sea competitiva es de "
            f"{proba_recom*100:.1f}%. Tus condiciones parecen alineadas con propiedades exitosas."
        )
    else:
        texto_rec = "✖ No recomendada"
        texto_rec_detalle = (
            f"La probabilidad estimada de competitividad es de {proba_recom*100:.1f}%. "
            "Puedes considerar ajustar el precio, mejorar tu tasa de respuesta o disponibilidad."
        )

    return texto_precio, texto_precio_detalle, texto_rec, texto_rec_detalle


# ----------------------------------------------------
# 5. Callbacks – Visualizaciones
# ----------------------------------------------------
@app.callback(
    Output("graf-roomtype", "figure"),
    Input("dd-room", "value"),
)
def grafico_roomtype(room_sel):
    fig = px.box(
        df,
        x="room_type",
        y="price",
        title="Distribución de precios por tipo de habitación",
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=400,
        showlegend=False,
    )
    return fig


@app.callback(
    Output("graf-barrio", "figure"),
    Input("dd-barrio", "value"),
)
def grafico_barrio(barrio_sel):
    df_b = df[df["neighbourhood_cleansed"] == barrio_sel]
    if df_b.empty:
        df_b = df.copy()
        titulo = "Precio por huésped (todos los barrios)"
    else:
        titulo = f"Precio por huésped en {barrio_sel}"

    df_plot = (
        df_b.groupby("room_type")["price_per_guest"]
        .mean()
        .reset_index()
        .sort_values("price_per_guest", ascending=False)
    )

    fig = px.bar(
        df_plot,
        x="room_type",
        y="price_per_guest",
        title=titulo,
        labels={"room_type": "Tipo de habitación", "price_per_guest": "Precio por huésped"},
    )
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=400, showlegend=False)
    return fig


@app.callback(
    Output("graf-acc-price", "figure"),
    Input("dd-room", "value"),
)
def grafico_acc_price(room_sel):
    df_r = df[df["room_type"] == room_sel]
    if df_r.empty:
        df_r = df.copy()
        titulo = "Capacidad vs precio (todos los tipos de habitación)"
    else:
        titulo = f"Capacidad vs precio para {room_sel}"

    fig = px.scatter(
        df_r,
        x="accommodates",
        y="price",
        title=titulo,
        opacity=0.6,
        labels={"accommodates": "Capacidad (huéspedes)", "price": "Precio"},
    )
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=400, showlegend=False)
    return fig


# ----------------------------------------------------
# 6. Ejecutar la app
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)

