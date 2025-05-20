# -*- coding: utf-8 -*-
"""
Created on Tue May 20 09:09:01 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from PIL import Image

# -------------------------------
# Configuración inicial
# -------------------------------
st.set_page_config(page_title="Evaluación de Políticas de Seguridad", layout="wide")

# -------------------------------
# Introducción y contexto
# -------------------------------
st.title("📊 Evaluación Inteligente de Políticas de Seguridad Pública")
st.markdown("Desarrollada por **Javier Horacio Pérez Ricárdez**, Doctorando en IA")

with st.expander("ℹ️ Acerca de esta herramienta", expanded=True):
    st.markdown("""
    Esta herramienta demuestra capacidades técnicas para evaluar políticas de seguridad pública mediante:
    - Análisis de presupuestos, beneficiarios y tasas de delitos
    - Modelado predictivo para identificar tendencias y relaciones clave
    - Detección de subejecuciones presupuestarias
    - Comparación con benchmarks internacionales
    
    **Objetivo:** Apoyar la toma de decisiones en políticas de seguridad mediante análisis cuantitativos y cualitativos,
    alineado con los estándares de evaluación del sector público.
    """)

# -------------------------------
# Nueva sección: Fortalecimiento Institucional
# -------------------------------
st.subheader("🏛️ Fortalecimiento Institucional y Control de Confianza")
with st.expander("🔍 Análisis Cualitativo y Buenas Prácticas", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Análisis de Documentos Normativos:**
        - Revisión sistemática de marcos legales y normativos
        - Evaluación de cumplimiento con estándares internacionales
        - Identificación de brechas regulatorias
        
        **Ejemplo de Buenas Prácticas Internacionales:**
        1. **Modelo Islandés de Prevención Juvenil**:
           - Enfoque comunitario integral
           - Reducción del 42% en consumo de sustancias
        2. **Policía de Proximidad (Portugal)**:
           - Enfoque humanista y preventivo
           - Aumento del 35% en percepción de seguridad
        """)
    
    with col2:
        st.markdown("""
        **Coordinación de Equipos Técnicos:**
        - Metodología ágil (Scrum) para equipos de 2-3 personas
        - Roles claros: Investigador principal, Analista de datos, Especialista en políticas
        - Revisión semanal de avances con matriz de responsabilidades
        
        **Alineación con Control de Confianza:**
        - Integración de evaluaciones periódicas de desempeño
        - Mecanismos de transparencia en la asignación de recursos
        - Sistema de monitoreo con indicadores de confiabilidad
        """)

# -------------------------------
# Nueva sección: Herramientas Técnicas
# -------------------------------
st.subheader("🛠️ Herramientas Técnicas Especializadas")
tab_tools1, tab_tools2, tab_tools3 = st.tabs(["RStudio", "QGIS", "Matrices de Indicadores"])

with tab_tools1:
    st.markdown("""
    **Ejemplo de Análisis en RStudio:**
    ```r
    # Modelo de regresión para tasa de delitos
    modelo <- lm(tasa_delitos ~ presupuesto + beneficiarios + factor(estado), data=datos)
    
    # Visualización interactiva
    library(ggplot2)
    ggplot(datos, aes(x=presupuesto, y=tasa_delitos, color=programa)) +
      geom_point() +
      geom_smooth(method="lm") +
      labs(title="Relación Presupuesto-Tasa de Delitos")
    ```
    """)
    st.image("https://raw.githubusercontent.com/rstudio/hex-stickers/master/PNG/RStudio.png", width=150)

with tab_tools2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Análisis Geoespacial en QGIS:**
        - Capas de delitos por zonas geográficas
        - Heatmaps de incidencia delictiva
        - Coropletas de eficiencia presupuestal
        """)
    with col2:
        # Mapa simulado de QGIS
        m_qgis = folium.Map(location=[23.6345, -102.5528], zoom_start=5)
        folium.Marker(
            location=[19.4326, -99.1332],
            popup="<b>CDMX</b><br>Alta incidencia delictiva<br>Presupuesto: $1,200M",
            icon=folium.Icon(color="red")
        ).add_to(m_qgis)
        st_folium(m_qgis, width=300, height=200)

with tab_tools3:
    st.markdown("""
    **Matriz de Indicadores de Resultados (MIR):**
    
    | Componente          | Indicador                          | Meta  | Línea Base | Fuente de Verificación |
    |---------------------|------------------------------------|-------|------------|------------------------|
    | Prevención delito   | % reducción tasa delitos           | 15%   | 100        | Estadísticas policiales |
    | Capacitación        | % policías certificados           | 80%   | 45%        | Registros SSP          |
    | Justicia cívica     | % casos resueltos en 30 días      | 75%   | 60%        | Sistema judicial       |
    """)
    st.markdown("**Proceso de implementación:**")
    st.progress(65)

# -------------------------------
# Generar datos simulados
# -------------------------------
@st.cache_data
def generar_datos():
    np.random.seed(42)
    estados = ['CDMX', 'Jalisco', 'Nuevo León', 'Chiapas', 'Yucatán']
    programas = ['Prevención del delito', 'Capacitación policial', 'Justicia cívica']
    años = list(range(2019, 2025))
    
    benchmarks = {
        'Prevención del delito': {'costo_efectividad': 0.85, 'impacto_promedio': 22.5},
        'Capacitación policial': {'costo_efectividad': 0.72, 'impacto_promedio': 18.3},
        'Justicia cívica': {'costo_efectividad': 0.65, 'impacto_promedio': 15.7}
    }
    
    data = []
    for estado in estados:
        for año in años:
            for programa in programas:
                presupuesto = np.random.uniform(50, 500)
                beneficiarios = presupuesto * np.random.uniform(10, 25)
                tasa_delitos = np.random.uniform(10, 50) - (beneficiarios / 5000)
                subejecucion = np.random.uniform(-20, 30)
                eficiencia = presupuesto / (tasa_delitos + 0.1)
                brecha_benchmark = tasa_delitos - benchmarks[programa]['impacto_promedio']
                
                data.append({
                    'Estado': estado,
                    'Año': año,
                    'Programa': programa,
                    'Presupuesto': round(presupuesto, 2),
                    'Beneficiarios': int(beneficiarios),
                    'Tasa_Delitos': round(tasa_delitos, 2),
                    'Subejecución': round(subejecucion, 2),
                    'Eficiencia': round(eficiencia, 2),
                    'Brecha_Benchmark': round(brecha_benchmark, 2),
                    'MIR_Cumplimiento': np.random.choice(['Alto', 'Medio', 'Bajo'], p=[0.3, 0.5, 0.2])
                })

    return pd.DataFrame(data), benchmarks

df, benchmarks = generar_datos()

# -------------------------------
# Filtros
# -------------------------------
st.sidebar.header("🎯 Filtros")
año = st.sidebar.selectbox("Año", sorted(df['Año'].unique()))
estado = st.sidebar.multiselect("Estado", df['Estado'].unique(), default=df['Estado'].unique())
programa = st.sidebar.multiselect("Programa", df['Programa'].unique(), default=df['Programa'].unique())

df_filtrado = df[(df['Año'] == año) &
                 (df['Estado'].isin(estado)) &
                 (df['Programa'].isin(programa))]

# -------------------------------
# Mostrar datos simulados
# -------------------------------
st.subheader("📄 Datos Simulados")
with st.expander("Ver datos completos"):
    st.dataframe(df_filtrado)

# -------------------------------
# KPI
# -------------------------------
st.subheader(f"📌 Indicadores Clave - Año {año}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Presupuesto Total (M)", f"${df_filtrado['Presupuesto'].sum():,.2f}")
col2.metric("Beneficiarios Totales", f"{df_filtrado['Beneficiarios'].sum():,}")
col3.metric("Tasa Prom. de Delitos", f"{df_filtrado['Tasa_Delitos'].mean():.2f}")
col4.metric("Eficiencia Promedio", f"{df_filtrado['Eficiencia'].mean():.2f}")

# -------------------------------
# Mapa geoespacial
# -------------------------------
st.subheader("🗺️ Análisis Geoespacial")

if not df_filtrado.empty:
    centro_mexico = [23.6345, -102.5528]
    m = folium.Map(location=centro_mexico, zoom_start=5)
    coordenadas = {
        'CDMX': [19.4326, -99.1332],
        'Jalisco': [20.6595, -103.3494],
        'Nuevo León': [25.5922, -99.9962],
        'Chiapas': [16.7569, -93.1292],
        'Yucatán': [20.7099, -89.0943]
    }
    
    for estado in df_filtrado['Estado'].unique():
        datos_estado = df_filtrado[df_filtrado['Estado'] == estado]
        tasa_promedio = datos_estado['Tasa_Delitos'].mean()
        presupuesto_total = datos_estado['Presupuesto'].sum()
        
        folium.Marker(
            location=coordenadas.get(estado, centro_mexico),
            popup=f"<b>{estado}</b><br>Tasa delitos: {tasa_promedio:.1f}<br>Presupuesto: ${presupuesto_total:,.0f}",
            tooltip=estado
        ).add_to(m)
    
    st_folium(m, width=700, height=400)

# -------------------------------
# Gráficos
# -------------------------------
st.subheader("📈 Análisis Presupuestal")
tab1, tab2, tab3 = st.tabs(["Presupuesto por Estado", "Relación Presupuesto vs. Delitos", "Benchmarks Internacionales"])

with tab1:
    fig1 = px.bar(df_filtrado, x='Estado', y='Presupuesto', color='Programa', 
                  barmode='group', title="Distribución de Presupuesto por Estado y Programa")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.scatter(df_filtrado, x='Presupuesto', y='Tasa_Delitos', color='Estado', 
                      size='Beneficiarios', hover_data=['Programa', 'Eficiencia'],
                      title="Relación entre Presupuesto y Tasa de Delitos")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Comparación con Estándares Internacionales")
    bench_df = pd.DataFrame(benchmarks).T.reset_index()
    bench_df.columns = ['Programa', 'Costo-Efectividad', 'Impacto Promedio']
    
    fig_bench = px.bar(bench_df, x='Programa', y='Impacto Promedio',
                       title="Impacto Promedio de Programas (Benchmark Internacional)")
    st.plotly_chart(fig_bench, use_container_width=True)
    
    st.markdown("**Brecha respecto a benchmark internacional:**")
    fig_brecha = px.box(df_filtrado, x='Programa', y='Brecha_Benchmark',
                        points="all", title="Diferencia entre tasa local y benchmark internacional")
    st.plotly_chart(fig_brecha, use_container_width=True)

# -------------------------------
# Modelo predictivo
# -------------------------------
st.subheader("🤖 Modelo Predictivo de Tasa de Delitos")

with st.expander("🔍 Explicación del Modelo", expanded=False):
    st.markdown("""
    **Modelo de Regresión Lineal:**
    - Predice la tasa de delitos en función del presupuesto y número de beneficiarios.
    - **R² (Coeficiente de determinación):** Mide qué porcentaje de la variación en la tasa de delitos es explicada por el modelo.
    - **MSE (Error Cuadrático Medio):** Promedio de los errores al cuadrado.
    - **MAE (Error Absoluto Medio):** Promedio de errores absolutos.
    """)

X = df_filtrado[['Presupuesto', 'Beneficiarios']]
y = df_filtrado['Tasa_Delitos']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.write("### 📊 Métricas del Modelo")
    metrics_df = pd.DataFrame({
        'Métrica': ['R² (Explicación)', 'MSE (Error Cuadrático)', 'MAE (Error Absoluto)'],
        'Valor': [r2, mse, mae],
        'Interpretación': [
            f"{r2*100:.1f}% de variación explicada",
            f"Error promedio: {mse:.2f} (unidades²)",
            f"Error promedio: {mae:.2f} unidades"
        ]
    })
    st.dataframe(metrics_df)

with col2:
    st.write("### 🔍 Coeficientes del Modelo")
    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coeficiente': modelo.coef_
    })
    st.dataframe(coef_df)

# -------------------------------
# Nueva sección: Visualización para Seguridad Pública
# -------------------------------
st.subheader("🚨 Visualización Especializada en Seguridad Pública")

# Datos simulados de tipos de delito
delitos_data = pd.DataFrame({
    'Tipo': ['Robo', 'Homicidio', 'Violencia familiar', 'Fraude', 'Narcomenudeo'],
    'Cantidad': [1250, 320, 780, 450, 210],
    'Tendencia': ['↑ 12%', '↓ 5%', '↑ 8%', '↑ 25%', '↓ 15%']
})

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.markdown("**Distribución de Delitos por Tipo**")
    fig_delitos = px.pie(delitos_data, values='Cantidad', names='Tipo',
                         hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_delitos, use_container_width=True)

with col_viz2:
    st.markdown("**Tendencias de Incidencia Delictiva**")
    fig_tendencia = px.bar(delitos_data, x='Tipo', y='Cantidad', color='Tendencia',
                           color_discrete_map={'↑ 12%': 'red', '↓ 5%': 'green', '↑ 8%': 'red',
                                              '↑ 25%': 'red', '↓ 15%': 'green'})
    st.plotly_chart(fig_tendencia, use_container_width=True)

# -------------------------------
# Nueva sección: Entrevistas Simuladas
# -------------------------------
st.subheader("🎤 Análisis Cualitativo - Entrevistas Simuladas")

with st.expander("📝 Metodología de Análisis de Entrevistas", expanded=False):
    st.markdown("""
    **Proceso de sistematización:**
    1. Transcripción y codificación de entrevistas (NVivo/Atlas.ti)
    2. Identificación de categorías temáticas emergentes
    3. Análisis de discurso y patrones recurrentes
    
    **Ejemplo de fragmento codificado:**
    > _"En la implementación del programa X encontramos resistencia al cambio por parte de los agentes..."_
    - Códigos aplicados: [Resistencia al cambio], [Cultura organizacional], [Capacitación insuficiente]
    """)

tab_ent1, tab_ent2 = st.tabs(["Percepciones de Efectividad", "Retos Identificados"])

with tab_ent1:
    st.markdown("""
    **Percepciones de actores clave:**
    - 65% considera los programas "efectivos pero mejorables"
    - 22% los califica como "muy efectivos"
    - 13% los considera "poco efectivos"
    
    **Citas relevantes:**
    > _"La capacitación policial ha mejorado, pero necesitamos más recursos para equipamiento"_
    (Comisario, Estado de México)
    
    > _"Los programas de prevención llegan tarde, deberían enfocarse en edades más tempranas"_
    (Directora de ONG local)
    """)

with tab_ent2:
    st.markdown("""
    **Principales retos identificados:**
    1. Coordinación interinstitucional (mencionado en 78% de entrevistas)
    2. Continuidad de programas (65%)
    3. Medición de impacto real (57%)
    
    **Recomendaciones emergentes:**
    - Fortalecer sistemas de monitoreo y evaluación
    - Mayor participación comunitaria en diseño de programas
    - Articulación con políticas sociales más amplias
    """)