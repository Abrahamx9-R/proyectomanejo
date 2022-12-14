#bibliotecas a utilizar
import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd
import math
import plotly.express as px

st.set_page_config(
    page_title="Proyecto",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.markdown("# Modelo Poisson para predicción de partidos de futbol")


def load_lottieurl(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_hello = load_lottieurl("images/proyecto.json")

st_lottie(lottie_hello,speed = 1, reverse=False,loop=True,quality="low",height=600,width=None,key=None,)


st.markdown("El modelo poisson es una herramienta de la probabilidad que tiene multiples uso en la medición de vida util de un evento, no obstante, esta distribucion también es usada para teimpos de espera en problemas de procesos estocasticos, en este proyecto se utilizará la distribución Poisson como herramienta de medición de calculo para obtener la probabilidad de que un equipo anote $X$ determinada cantidad de goles. ")
st.latex(r"f_x(X) = \frac{e^{-\lambda}\lambda^{x}}{x!} = P[X = x]")

data = pd.read_csv("Info_clean_2.csv")

st.markdown("Para este problema usaremos el dataset Info_clean_2.csv, que se puede descargar en el siguiente link")


with open("Info_clean_2.xlsx", "rb") as file:
    btn = st.download_button(
            label="Descarga dataset",
            data=file,
            file_name="Info_clean_2.xlsx"
          )

st.markdown("Este dataset tiene la siguiente informacion")

st.dataframe(data.tail(10))
#definicion de columnas con las cuales trabajar
columns_work = ['idPartido' ,'temporada' ,'jornada' ,'EquipoLocal' ,'EquipoVisitante' ,'Goles_L' ,'Goles_V']
soccer = data.copy()
soccer = soccer[columns_work]

st.markdown("Cual es la temporada que tomaremos por referencia")
referencia = st.selectbox("Temporada",soccer['temporada'].unique())


#Nota: la modelación de este proyecto esta basada en que se tomara un torneo anterior para modelar todo el torneo en tiempo presente. Esto se decidio despues de un analsisis y serie de pruebas con otros modelos 
st.markdown("Cual es la temporada que quieres predecir")
temporadas = soccer['temporada'].unique().tolist()
lista = []
for temporada in temporadas:
    if temporada != referencia:
        lista.append(temporada)
prediccion = st.selectbox("Predicción",lista)
# filtrado, en la temporada (t-1)
mask = (soccer['temporada'] == referencia)
soccer = soccer[mask]
# particion de la data y configuracion por local y visitante
locales = soccer.copy() 
del locales['EquipoVisitante']
locales = locales.rename(columns = {'EquipoLocal':'equipo',
                                    'Goles_L':'goles_favor',
                                    'Goles_V':'goles_contra'
                                    })
visitantes = soccer.copy() 
del visitantes['EquipoLocal']
visitantes = visitantes.rename(columns = {'EquipoVisitante':'equipo',
                                    'Goles_L':'goles_contra',
                                    'Goles_V':'goles_favor'
                                    })
#promedio general de goles anotados como local y visitante
prom_gol_l = soccer['Goles_L'].mean() 
prom_gol_v = soccer['Goles_V'].mean()

st.write("El promedio general de goles anotados como local y visitante en el temporada ",referencia,"es, visitantes: ",prom_gol_v," local ", prom_gol_l)
#promedio de goles por equipo del torneo pasado LOCALES
soccer_l_prom = locales.groupby(['equipo']).agg({'goles_favor':'mean' ,'goles_contra':'mean'})
soccer_l_prom = soccer_l_prom.rename(columns = {'goles_favor':'prom_gf_l',
                                    'goles_contra':'prom_gc_l',
                                    })
#VISITANTES
soccer_v_prom = visitantes.groupby(['equipo']).agg({'goles_favor':'mean' ,'goles_contra':'mean'})
soccer_v_prom = soccer_v_prom.rename(columns = {'goles_favor':'prom_gf_v',
                                    'goles_contra':'prom_gc_v',
                                    })
#Factor de ataque: una especie de pondracion para los goles anotados, es decir, es un promedio ponderado de los goles que un equipo memte como local/visitante entre el promedio general 
#LOCALES
#factor de ataque
soccer_l_prom['f_a_l'] = soccer_l_prom['prom_gf_l'] / prom_gol_l
#factor de defensa
soccer_l_prom['f_d_l'] = soccer_l_prom['prom_gc_l'] / prom_gol_v
#VISITANTES
#factor de ataque
soccer_v_prom['f_a_v'] = soccer_v_prom['prom_gf_v'] / prom_gol_v
#factor de defensa 
soccer_v_prom['f_d_v'] = soccer_v_prom['prom_gc_v'] / prom_gol_l

# ABT (analytical Base Table): tabla base para poder aplicar el modelo matematico que resolverá el promedio
abt = soccer_l_prom.merge(soccer_v_prom, on = ['equipo'])

st.markdown("A continuacioon mostramos una tabla base para poder aplicar el modelo matematico que resolvera el promedio")
st.dataframe(abt.head(2))

#los torneos que se buscan predecir son aquellos que son los complementarios a un año calnedario futbol, es decir, el torneo en tiempo t
#los resultados que buscamos predecir son los del torneo 'actual'
# por eso ahora tomaremos un data set con el torneo complementario del anio
soccer_to_pred = data.copy()
soccer_to_pred = soccer_to_pred[columns_work]
mask_pred      = (soccer_to_pred['temporada'] == prediccion)
soccer_to_pred = soccer_to_pred[mask_pred]

merge_1 =  soccer_to_pred.merge(abt, left_on = 'EquipoLocal', right_on = 'equipo')

columns_pred_work = ['idPartido', 'temporada',
		 'jornada', 'EquipoLocal', 'EquipoVisitante',
		 'Goles_L', 'Goles_V', 'f_a_l', 'f_d_l'] #,'f_a_v', 'f_d_v'
#unimos las tablas de la condicion de visitante y local para tener 
# todos los factores que implican al modelado
merge_1 = merge_1[columns_pred_work]
# uniendo la data de los visitantes
merge_2 = merge_1.merge(abt, left_on = 'EquipoVisitante', right_on = 'equipo')

columns_pred_work = ['idPartido', 'temporada',
		 'jornada', 'EquipoLocal', 'EquipoVisitante',
		 'Goles_L', 'Goles_V', 'f_a_l_x', 'f_d_l_x', 'f_a_v', 'f_d_v']

merge_2 = merge_2[columns_pred_work]
#Calculo de las lambdas para los parametros de la poisson: en este proyecto el parametro lambda es el promedio ponderado de los goles por equipo
lambdas = merge_2.copy()
# el factor de ataque local * factor de defensa de la visita * promedio de goles formaran la lambda
lambdas['lambda_l'] = lambdas['f_a_l_x'] * lambdas['f_d_v'] * prom_gol_l
lambdas['lambda_v'] = lambdas['f_a_v'] * lambdas['f_d_l_x'] * prom_gol_v


#maximo y minimo de goles del torneo anterior
# para establecer como rango de la poisson, es decir,
# por donde debe de correr la x
max_goals = soccer['Goles_L'].max()
min_goals = soccer['Goles_L'].min()

st.write("El maximo y minimo de goles del torneo de la temporada ",referencia," son ",min_goals,max_goals)

def pois_goals_p(lambd:float ,num_goals:int) -> float:
    ''' Esta función calcula la probabilidad de meter 
    num_goals en un partido con el parametro lambda dado
    
    >>> pois_goals_p(3, 4)
    0.16803135574154082

    esto se puede traducir como:
    la probabilidad de meter 4 goles dado que el 
    promedio de goles del equipo es de 3
    '''
    return ( ( (math.exp(-lambd)) * (lambd ** num_goals) ) / (math.factorial(num_goals)) )

#probabilidades de meter n goles para el equipo local
for i in range(min_goals ,max_goals + 1):
    lambdas[f'P_GL({i})'] = lambdas['lambda_l'].apply(pois_goals_p, num_goals = i)

#probabilidades de meter n goles para el equipo visitante
for i in range(min_goals ,max_goals + 1):
    lambdas[f'P_GV({i})'] = lambdas['lambda_v'].apply(pois_goals_p, num_goals = i)

pre_final = lambdas.copy()
P_G  = []
for i in range(min_goals,max_goals+1):
    P_G.append(f"P_GL({i})")
for i in range(min_goals,max_goals+1):
    P_G.append(f"P_GV({i})")
pre_final = pre_final[['idPartido', 'temporada', 'jornada', 'EquipoLocal', 'EquipoVisitante','Goles_L', 'Goles_V']+P_G]

# inicializamos nuevas variables para establecer la proba de que gane el local
# visitante o haya empate
pre_final['P_Local'] = 0
pre_final['P_Visit'] = 0
pre_final['P_Empat'] = 0

# El siguiente ´while´ es para multiplicar todas las probas marginales de que un equipo gane, es decir, la proba de que el local gane es: Todos aquellos resultados en los cuales el local anota más goles que el visitante, es por es que, se tiene que multiplicar
# P[Local] = \sum_{i<j}^n P[Goles_Local(i)] > P[Goles_Visitante(j)] 

while max_goals > min_goals:
    pre_final['P_Local'] = pre_final['P_Local'] + (pre_final[f'P_GL({max_goals})'] * pre_final[f'P_GV({max_goals - 1})'])
    pre_final['P_Visit'] = pre_final['P_Visit'] + (pre_final[f'P_GV({max_goals})'] * pre_final[f'P_GL({max_goals - 1})'])
    pre_final['P_Empat'] = pre_final['P_Empat'] + (pre_final[f'P_GL({max_goals})'] * pre_final[f'P_GV({max_goals})'])

    max_goals -= 1
    
final = pre_final.copy()

final = final[['temporada', 'jornada', 'EquipoLocal', 'EquipoVisitante',
       'Goles_L', 'Goles_V', 'P_Local', 'P_Visit', 'P_Empat']]

final['preds'] = 'SA'

# asignacion de variable gana local
mask_local = (final['P_Local'] > final['P_Visit'])
final.loc[mask_local, 'preds'] = 'L'

# asignacion de variable gana visitante
mask_visit = (final['P_Local'] < final['P_Visit'])
final.loc[mask_visit, 'preds'] = 'V'

# asignacion de variable empate
mask_empt = (( (final['P_Empat'] > final['P_Local']) & (final['P_Empat'] > final['P_Visit']) ) 
            | ((abs(final['P_Local'] - final['P_Visit'])) < 0.05) )
final.loc[mask_empt, 'preds'] = 'E'

# etiqueta de resultados reales
final['result'] = 'SA'

res_local = ( final['Goles_L'] > final['Goles_V'] )
res_visit = ( final['Goles_L'] < final['Goles_V'] )
res_empat = ( final['Goles_L'] == final['Goles_V'] )

final.loc[res_local, 'result'] = 'L'
final.loc[res_visit, 'result'] = 'V'
final.loc[res_empat, 'result'] = 'E'

st.write("Este algoritmo hizo la siguiente prediccion de victorias visitantes, locales o empates")

st.write(final['preds'].value_counts())
st.write("Y los resultados son los siguientes") 
st.write(final['result'].value_counts())

grafica = pd.DataFrame([final['result'].value_counts(),final['preds'].value_counts()])


import plotly.express as px
long_df = px.data.medals_long()
 
fig = px.bar(grafica.transpose(),
            title = "Resultados y predicciones")

st.plotly_chart(fig)
#El modelo al menos en proporciones las respeta, pero nos interesa saber, cuantas predicciones correctas se hicieron
acuracy = final['preds'] == final['result']



x = list(acuracy.value_counts())

precision = x[1] / sum(x)
st.write(f'La precision del modelo es de {round(precision*100,2)} %')


referencia = st.selectbox("Jornada",sorted(final["jornada"].unique().tolist()))

resultadoJornada = (final["jornada"] == referencia)
resultadoJornada = final[resultadoJornada]
st.markdown("A continuacion se puede ver los resultados por jornada")
st.dataframe(resultadoJornada)


