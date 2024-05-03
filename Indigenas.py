import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import altair as alt
from tqdm import tqdm
from tabulate import tabulate


# Define the model function
def model_trabajo(t, y, params):
    trabajo_inf_salario_min_indigenas, trabajo_sup_salario_min_indigenas = y
    # Unpacking parameters
    tasa_de_gasto_educacion, discriminacion, costo_por_año_educacion_promedio, *fixed_params = params
    temporalidad_politica_publica, becas_personas_indigenas, ingreso_sup_salario_min_promedio, \
    ingreso_inf_salario_min_promedio, politica_educativa, aumento_contratacion_escolaridad, \
    tasa_de_contratacion_promedio, tasa_de_desempleo_promedio, costo_minimo_de_vida = fixed_params
    
    costo_por_año_educacion = max(0, costo_por_año_educacion_promedio - (politica_educativa / temporalidad_politica_publica))
    gastos_electivos = (ingreso_sup_salario_min_promedio * trabajo_sup_salario_min_indigenas + ingreso_inf_salario_min_promedio * trabajo_inf_salario_min_indigenas) - costo_minimo_de_vida
    inversion_en_educacion = (gastos_electivos * tasa_de_gasto_educacion) + (becas_personas_indigenas * (trabajo_inf_salario_min_indigenas + trabajo_sup_salario_min_indigenas))
    años_de_escolaridad = inversion_en_educacion / costo_por_año_educacion
    años_de_escolaridad_por_persona = años_de_escolaridad / (trabajo_sup_salario_min_indigenas + trabajo_inf_salario_min_indigenas)
    
    base_contratacion_rate = tasa_de_contratacion_promedio + (aumento_contratacion_escolaridad * años_de_escolaridad_por_persona)
    tasa_de_contratacion = min(1, max(0, base_contratacion_rate * (1 - discriminacion)))
    tasa_de_desempleo = min(1, max(0, tasa_de_desempleo_promedio + (tasa_de_contratacion_promedio * discriminacion)))
    
    contratacion = tasa_de_contratacion * trabajo_inf_salario_min_indigenas
    desempleo = trabajo_sup_salario_min_indigenas * tasa_de_desempleo
    
    dtrabajo_inf_salario_min_indigenas = desempleo - contratacion
    dtrabajo_sup_salario_min_indigenas = contratacion - desempleo
    
    return [dtrabajo_inf_salario_min_indigenas, dtrabajo_sup_salario_min_indigenas]

# Fixed parameters
fixed_parameters = [
    10,  # temporalidad_politica_publica
    0,   # becas_personas_indigenas
    73311,  # ingreso_sup_salario_min_promedio
    54900,  # ingreso_inf_salario_min_promedio
    0,   # politica_educativa
    0.025 * 0.05,  # aumento_contratacion_escolaridad
    0.025,  # tasa_de_contratacion_promedio
    0.05,  # tasa_de_desempleo_promedio
    39000  # costo_minimo_de_vida
]
# Define parameters and ranges
step_gasto = 0.01
step_discrim = 0.01
step_costo_educ = 100
gasto_educ_range = np.arange(0.01, 0.21 + step_gasto, step_gasto)
discrim_range = np.arange(0.20, 0.40 + step_discrim, step_discrim)
costo_educ_range = np.arange(1, 5001 + step_costo_educ, step_costo_educ)

@st.cache_data
def compute_all_simulations():
    all_params = list(itertools.product(gasto_educ_range, discrim_range, costo_educ_range))
    results = {}
    fixed_parameters = [10, 0, 73311, 54900, 0, 0.025 * 0.05, 0.025, 0.05, 39000]

    for params in all_params:
        solution = solve_ivp(model_trabajo, [0, 50], [6660000, 2340000], t_eval=np.linspace(0, 50, 101), args=(list(params) + fixed_parameters,))
        results[tuple(np.round(params, decimals=3))] = pd.DataFrame({
            'Time': solution.t,
            'Empleos inferiores al salario minimo': solution.y[0],
            'Empleos superiores al salario minimo': solution.y[1]
        }).set_index('Time')
    
    return results

# Load cached simulations
results = compute_all_simulations()




# Streamlit app setup
st.title("Simulación de Dinámicas de Empleo para Poblaciones Indígenas")

st.header("Introducción")
st.write("Esta aplicación tiene como objetivo simular las dinámicas de empleo para las poblaciones indígenas, teniendo en cuenta diversos parámetros económicos y sociales. A través de esta herramienta, podrás explorar cómo diferentes combinaciones de parámetros afectan los niveles de empleo a lo largo del tiempo.")

st.header("Explicación de los Parámetros")

# Define parameters and their descriptions
parameters = {
"Temporalidad Política Pública": {
"Descripción": "Placeholder text for Temporalidad Política Pública.",
"Valor": 10
},
"Becas Personas Indígenas": {
"Descripción": "Placeholder text for Becas Personas Indígenas.",
"Valor": 0
},
"Ingreso Superior al Salario Mínimo Promedio": {
"Descripción": "Placeholder text for Ingreso Superior al Salario Mínimo Promedio.",
"Valor": 73311
},
"Ingreso Inferior al Salario Mínimo Promedio": {
"Descripción": "Placeholder text for Ingreso Inferior al Salario Mínimo Promedio.",
"Valor": 54900
},
"Política Educativa": {
"Descripción": "Placeholder text for Política Educativa.",
"Valor": 0
},
"Aumento de Contratación por Escolaridad": {
"Descripción": "Placeholder text for Aumento de Contratación por Escolaridad.",
"Valor": 0.00125
},
"Tasa de Contratación Promedio": {
"Descripción": "Placeholder text for Tasa de Contratación Promedio.",
"Valor": 0.025
},
"Tasa de Desempleo Promedio": {
"Descripción": "Placeholder text for Tasa de Desempleo Promedio.",
"Valor": 0.05
},
"Costo Mínimo de Vida": {
"Descripción": "Placeholder text for Costo Mínimo de Vida.",
"Valor": 39000
}
}
headers = ["Variable", "Descripción", "Valor"]

table_data = [[variable, data["Descripción"], data["Valor"]] for variable, data in parameters.items()]

# Conver thte data to a pandas DataFrame
parameters_df = pd.DataFrame(table_data, columns=headers)

# Display the table in the Streamlit app
st.table(parameters_df)

st.header("Código de la Simulación:")
with st.expander("Ver código"):
    st.code('''
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import altair as alt
from tqdm import tqdm

# Define the model function
def model_trabajo(t, y, params):
    trabajo_inf_salario_min_indigenas, trabajo_sup_salario_min_indigenas = y
    # Unpacking parameters
    tasa_de_gasto_educacion, discriminacion, costo_por_año_educacion_promedio, *fixed_params = params
    temporalidad_politica_publica, becas_personas_indigenas, ingreso_sup_salario_min_promedio, \
    ingreso_inf_salario_min_promedio, politica_educativa, aumento_contratacion_escolaridad, \
    tasa_de_contratacion_promedio, tasa_de_desempleo_promedio, costo_minimo_de_vida = fixed_params
    
    costo_por_año_educacion = max(0, costo_por_año_educacion_promedio - (politica_educativa / temporalidad_politica_publica))
    gastos_electivos = (ingreso_sup_salario_min_promedio * trabajo_sup_salario_min_indigenas + ingreso_inf_salario_min_promedio * trabajo_inf_salario_min_indigenas) - costo_minimo_de_vida
    inversion_en_educacion = (gastos_electivos * tasa_de_gasto_educacion) + (becas_personas_indigenas * (trabajo_inf_salario_min_indigenas + trabajo_sup_salario_min_indigenas))
    años_de_escolaridad = inversion_en_educacion / costo_por_año_educacion
    años_de_escolaridad_por_persona = años_de_escolaridad / (trabajo_sup_salario_min_indigenas + trabajo_inf_salario_min_indigenas)
    
    base_contratacion_rate = tasa_de_contratacion_promedio + (aumento_contratacion_escolaridad * años_de_escolaridad_por_persona)
    tasa_de_contratacion = min(1, max(0, base_contratacion_rate * (1 - discriminacion)))
    tasa_de_desempleo = min(1, max(0, tasa_de_desempleo_promedio + (tasa_de_contratacion_promedio * discriminacion)))
    
    contratacion = tasa_de_contratacion * trabajo_inf_salario_min_indigenas
    desempleo = trabajo_sup_salario_min_indigenas * tasa_de_desempleo
    
    dtrabajo_inf_salario_min_indigenas = desempleo - contratacion
    dtrabajo_sup_salario_min_indigenas = contratacion - desempleo
    
    return [dtrabajo_inf_salario_min_indigenas, dtrabajo_sup_salario_min_indigenas]

# Fixed parameters
fixed_parameters = [
    10,  # temporalidad_politica_publica
    0,   # becas_personas_indigenas
    73311,  # ingreso_sup_salario_min_promedio
    54900,  # ingreso_inf_salario_min_promedio
    0,   # politica_educativa
    0.025 * 0.05,  # aumento_contratacion_escolaridad
    0.025,  # tasa_de_contratacion_promedio
    0.05,  # tasa_de_desempleo_promedio
    39000  # costo_minimo_de_vida
]
# Define parameters and ranges
step_gasto = 0.01
step_discrim = 0.01
step_costo_educ = 100
gasto_educ_range = np.arange(0.01, 0.21 + step_gasto, step_gasto)
discrim_range = np.arange(0.20, 0.40 + step_discrim, step_discrim)
costo_educ_range = np.arange(1, 5001 + step_costo_educ, step_costo_educ)

@st.cache_data
def compute_all_simulations():
    all_params = list(itertools.product(gasto_educ_range, discrim_range, costo_educ_range))
    results = {}
    fixed_parameters = [10, 0, 73311, 54900, 0, 0.025 * 0.05, 0.025, 0.05, 39000]

    for params in all_params:
        solution = solve_ivp(model_trabajo, [0, 50], [6660000, 2340000], t_eval=np.linspace(0, 50, 101), args=(list(params) + fixed_parameters,))
        results[tuple(np.round(params, decimals=3))] = pd.DataFrame({
            'Time': solution.t,
            'Empleos inferiores al salario minimo': solution.y[0],
            'Empleos superiores al salario minimo': solution.y[1]
        }).set_index('Time')
    
    return results

# Load cached simulations
results = compute_all_simulations()

            
def generate_chart(fixed_param1, fixed_param2, varying_param_range, fixed_indices, varying_param_index, param_name):
    all_lines = []  # List to store all the lines for the chart

    for value in varying_param_range:
        params = [0, 0, 0]  # Initialize with placeholder list
        params[varying_param_index] = value
        params[fixed_indices[0]] = fixed_param1
        params[fixed_indices[1]] = fixed_param2
        
        # Ensure parameters are rounded to match the simulation grid exactly
        rounded_params = tuple(round(p, 3) if i == 0 else round(p, 2) if i == 1 else int(p) for i, p in enumerate(params))
        
        scenario_data = results[rounded_params]
        normalized_data = scenario_data.div(scenario_data.sum(axis=1), axis=0)
        normalized_data['Parameter Value'] = value  # Add the parameter value as a column for coloring
        
        all_lines.append(normalized_data.reset_index())

    # Concatenate all lines into a single DataFrame
    all_data = pd.concat(all_lines)
    
    # Create a color scale that varies from red to green
    color_scale = alt.Scale(domain=(min(varying_param_range), max(varying_param_range)), range=['red', 'green'])

    # Create the chart using all the data
    chart = alt.Chart(all_data).mark_line().encode(
        x='Time',
        y=alt.Y('Empleos superiores al salario minimo', axis=alt.Axis(title='Proporción de personas con Empleos superiores al salario minimo')),
        color=alt.Color('Parameter Value', scale=color_scale),
        tooltip=['Time', 'Empleos superiores al salario minimo', 'Parameter Value']
    ).properties(
        title=f"Simulación en el tiempo cambiando: {param_name}"
    ).interactive()

    return chart

# Identify the extreme scenarios
max_employment = max(final_values, key=lambda x: final_values[x]['Empleos inferiores al salario minimo'])
min_employment = min(final_values, key=lambda x: final_values[x]['Empleos inferiores al salario minimo'])

# Access the specific dataframes for maximum and minimum scenarios
max_df = results[max_employment]
min_df = results[min_employment]

# Normalize data for plotting
max_normalized = max_df.div(max_df.sum(axis=1), axis=0)
min_normalized = min_df.div(min_df.sum(axis=1), axis=0)

# Combine the data for plotting
extreme_data = pd.concat([max_normalized.reset_index(), min_normalized.reset_index()], keys=['Max', 'Min'])

# Add a new column to indicate the scenario
extreme_data['Scenario'] = extreme_data.index.get_level_values(0)

# Create the chart
extreme_chart = alt.Chart(extreme_data).mark_line().encode(
    x='Time',
    y=alt.Y('Empleos inferiores al salario minimo', axis=alt.Axis(title='Proporción de personas con Empleos inferiores al salario minimo')),
    color='Scenario',
    tooltip=['Time', 'Empleos inferiores al salario minimo', 'Scenario']
).properties(
    title="Comparación de los escenarios extremos"
).interactive()
''', language='python')


st.header("Simulación de Dinámicas de Empleo para Poblaciones Indígenas")
# Define sliders with precise step values and rounding
selected_gasto = round(st.slider("Tasa de Gasto en Educación", 0.01, 0.21, 0.11, step_gasto), 2)
selected_discrim = round(st.slider("Nivel de Discriminación", 0.20, 0.40, 0.28, step_discrim), 2)
selected_costo_educ = int(st.slider("Costo Promedio por Año de Educación", 1, 5001, 1001, step_costo_educ))

# Accessing the user selection graph
user_selection = results[(selected_gasto, selected_discrim, selected_costo_educ)]

# Normalize data for graphing
normalized_selection = user_selection.div(user_selection.sum(axis=1), axis=0)

# Plotting
st.line_chart(normalized_selection)


# Function to generate charts for varying parameters
def generate_chart(fixed_param1, fixed_param2, varying_param_range, fixed_indices, varying_param_index, param_name):
    all_lines = []  # List to store all the lines for the chart

    for value in varying_param_range:
        params = [0, 0, 0]  # Initialize with placeholder list
        params[varying_param_index] = value
        params[fixed_indices[0]] = fixed_param1
        params[fixed_indices[1]] = fixed_param2
        
        # Ensure parameters are rounded to match the simulation grid exactly
        rounded_params = tuple(round(p, 3) if i == 0 else round(p, 2) if i == 1 else int(p) for i, p in enumerate(params))
        
        scenario_data = results[rounded_params]
        normalized_data = scenario_data.div(scenario_data.sum(axis=1), axis=0)
        normalized_data['Parameter Value'] = value  # Add the parameter value as a column for coloring
        
        all_lines.append(normalized_data.reset_index())

    # Concatenate all lines into a single DataFrame
    all_data = pd.concat(all_lines)
    
    # Create a color scale that varies from red to green
    color_scale = alt.Scale(domain=(min(varying_param_range), max(varying_param_range)), range=['red', 'green'])

    # Create the chart using all the data
    chart = alt.Chart(all_data).mark_line().encode(
        x='Time',
        y=alt.Y('Empleos superiores al salario minimo', axis=alt.Axis(title='Proporción de personas con Empleos superiores al salario minimo')),
        color=alt.Color('Parameter Value', scale=color_scale),
        tooltip=['Time', 'Empleos superiores al salario minimo', 'Parameter Value']
    ).properties(
        title=f"Simulación en el tiempo cambiando: {param_name}"
    ).interactive()

    return chart


# Generate and display charts for each parameter
st.header("Efecto de Variar los Parámetros Individualmente")
chart_gasto = generate_chart(selected_discrim, selected_costo_educ, gasto_educ_range, [1, 2], 0, "Tasa de Gasto en Educación")
st.altair_chart(chart_gasto)

chart_discrim = generate_chart(selected_gasto, selected_costo_educ, discrim_range, [0, 2], 1, "Nivel de Discriminación")
st.altair_chart(chart_discrim)

chart_costo_educ = generate_chart(selected_gasto, selected_discrim, costo_educ_range, [0, 1], 2, "Costo Promedio por Año de Educación")
st.altair_chart(chart_costo_educ)


st.header("Efecto de Variar los 3 Parámetros Simultáneamente")
# Aggregate results to find extremes
final_values = {params: df.iloc[-1] for params, df in results.items()}

# Identify the extreme scenarios
max_employment = max(final_values, key=lambda x: final_values[x]['Empleos inferiores al salario minimo'])
min_employment = min(final_values, key=lambda x: final_values[x]['Empleos inferiores al salario minimo'])

# Access the specific dataframes for maximum and minimum scenarios
max_df = results[max_employment]
min_df = results[min_employment]

# Normalize data for plotting
max_normalized = max_df.div(max_df.sum(axis=1), axis=0)
min_normalized = min_df.div(min_df.sum(axis=1), axis=0)

# Combine the data for plotting
extreme_data = pd.concat([max_normalized.reset_index(), min_normalized.reset_index()], keys=['Max', 'Min'])

# Add a new column to indicate the scenario
extreme_data['Scenario'] = extreme_data.index.get_level_values(0)

# Create the chart
extreme_chart = alt.Chart(extreme_data).mark_line().encode(
    x='Time',
    y=alt.Y('Empleos inferiores al salario minimo', axis=alt.Axis(title='Proporción de personas con Empleos inferiores al salario minimo')),
    color='Scenario',
    tooltip=['Time', 'Empleos inferiores al salario minimo', 'Scenario']
).properties(
    title="Comparación de los escenarios extremos"
).interactive()

# Display the chart in the Streamlit app
st.altair_chart(extreme_chart)

# Report on the parameter combinations
st.write(f"La combinación de parámetros que da el máximo empleo inferior al salario mínimo es: Tasa de Gasto en Educación: {max_employment[0]}, Nivel de Discriminación: {max_employment[1]}, Costo Promedio por Año de Educación: {max_employment[2]}")
st.write(f"La combinación de parámetros que da el mínimo empleo inferior al salario mínimo es: Tasa de Gasto en Educación: {min_employment[0]}, Nivel de Discriminación: {min_employment[1]}, Costo Promedio por Año de Educación: {min_employment[2]}")

# Report total number of simulations
st.write(f"Se realizaron {len(results)} simulaciones en total.")

st.header("Recursos Adicionales")
st.write("A continuación se encuentran algunos recursos adicionales relacionados con el tema:")
st.write("- [Informe sobre la situación laboral de las poblaciones indígenas](https://example.com/informe)")
st.write("- [Artículo sobre políticas públicas para el empleo indígena](https://example.com/articulo)")