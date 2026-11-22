# -*- coding: utf-8 -*-
"""Explorador de resultados TINTOlib. Ejecutar con streamlit run app.py"""
from __future__ import annotations
import os
import io
from zipfile import ZipFile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from analisis_core import (cargar, preparar_archivos_subidos, serie_lineas, ordenar, seleccionar_mejores,
    tabla_victorias, alertas, frontera_pareto, calidad, MODELOS_BASE, METODOS,
    ARQUITECTURAS, METRICAS, MAYOR_MEJOR, ARCHIVOS_ANALISIS, OBLIGATORIOS)

ROOT=Path(__file__).resolve().parent

# ZIP publico de ejemplo. Los CSV de usuarios siguen siendo independientes.
DEMO_ZIP = ROOT / 'demo' / 'Local_ejemplo.zip'
MAX_CSV_BYTES = 80 * 1024 * 1024


@st.cache_data(show_spinner='Cargando los resultados de ejemplo...')
def cargar_ejemplo_zip(ruta_zip: str, version_archivo: int):
    """Lee solo los CSV requeridos del ZIP incluido, sin extraerlo a disco.

    version_archivo invalida la cache cuando se reemplaza el ZIP.
    """
    del version_archivo  # Se utiliza en la clave de cache.
    grupos = {'local': {}, 'directo': {}, 'colab': {}}
    total_bytes = 0

    with ZipFile(ruta_zip, 'r') as archivo_zip:
        for entrada in archivo_zip.infolist():
            if entrada.is_dir():
                continue
            nombre = entrada.filename.replace('\\', '/')
            partes = [parte for parte in nombre.split('/') if parte and parte != '.']
            if '..' in partes or len(partes) < 2:
                continue
            ruta_rel = '/'.join(partes[-2:])
            ruta_can = next((r for r in ARCHIVOS_ANALISIS if r.casefold() == ruta_rel.casefold()), None)
            if ruta_can is None:
                continue
            prefijo = [p.casefold() for p in partes[:-2]]
            grupo = 'local' if 'local' in prefijo else 'colab' if 'colab' in prefijo else 'directo'
            if ruta_can in grupos[grupo]:
                raise ValueError(f'CSV duplicado en el ZIP de ejemplo: {ruta_can}')
            if entrada.file_size > MAX_CSV_BYTES or total_bytes + entrada.file_size > MAX_CSV_BYTES:
                raise ValueError('Los CSV del ZIP de ejemplo superan el limite de 80 MB.')
            total_bytes += entrada.file_size
            grupos[grupo][ruta_can] = archivo_zip.read(entrada)

    elegido = next((g for g in ('local', 'directo', 'colab') if OBLIGATORIOS.issubset(grupos[g])), None)
    if elegido is None:
        raise ValueError('El ZIP de ejemplo no contiene Maestros/Resultados_Globales_Por_Modelo.csv y Datasets/Tabla_Datasets.csv.')
    archivos = grupos[elegido]
    frames = cargar(archivos)
    return (*frames, len(archivos), sum(len(b) for b in archivos.values()) / (1024 ** 2))


COLORS=['#66D9E8','#A6E36A','#F7B86B','#B8A0FF','#FF799B','#5DB8FF','#E3CE79','#69CBB3','#D69CEC','#E99976','#99C7F2','#F2A6C4','#B4DD8D','#CBBAD7']
BG='#101114'; PANEL='#181A20'; INK='#F4F4F6'; MUTED='#A7A8AC'

st.set_page_config(page_title='TINTOlib | Explorador científico',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
/* Diseño minimalista: interfaz clara y gráficas oscuras */
.stApp {background:#FFFFFF;color:#1D1D1F;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.block-container {padding-top:1.8rem;padding-bottom:2.5rem;max-width:1660px}
h1,h2,h3,h4,p {color:#1D1D1F}
h1 {font-weight:650;letter-spacing:-.035em;font-size:2rem!important}
h2,h3 {letter-spacing:-.022em;font-weight:600}
[data-testid="stMetric"] {background:#F7F7F8;border:1px solid #E6E6E8;border-radius:15px;padding:15px}
[data-testid="stDataFrame"] {border:1px solid #E5E5E8;border-radius:12px;overflow:hidden}
[data-testid="stTabs"] button {color:#6E6E73;font-weight:500}
[data-testid="stTabs"] button[aria-selected="true"] {color:#111111!important}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:15px}
[data-testid="stAlert"] {border-radius:12px}
button[kind="primary"] {border-radius:10px}
[data-testid="stCaptionContainer"] {color:#74747A!important}
hr {border-color:#EDEDEF}
/* Plotly utiliza el ancho del contenedor Streamlit; no forzar max-content.
   El rango visible se administra con los controles de navegación. */
[data-testid="stPlotlyChart"] {min-width:0!important;max-width:100%!important}
/* Controles compactos */
.st-key-panel_explorar [data-testid="stVerticalBlock"] {gap:.45rem}
.st-key-panel_explorar label {font-size:.82rem}
.st-key-panel_explorar [data-testid="stWidgetLabel"] {margin-bottom:0}
.st-key-panel_explorar [data-baseweb="select"] > div {min-height:36px}

</style>''',unsafe_allow_html=True)


def chart_style(fig,title=None,xlabel=None,ylabel=None,legend=True):
    fig.update_layout(template='plotly_dark',paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(color=INK,size=12),height=610,hovermode='closest',
        margin=dict(l=55,r=220,t=60,b=120),title=title,title_font=dict(color='#FFFFFF'),
        legend=dict(orientation='v',y=1,x=1.02,xanchor='left',yanchor='top',bgcolor='rgba(0,0,0,0)',font=dict(color='#F4F4F6',size=10)),showlegend=legend,
        xaxis=dict(showgrid=True,gridcolor='#34363D',zeroline=False,title=xlabel,linecolor='#B8BAC2',tickfont=dict(color='#E2E2E7')),
        yaxis=dict(showgrid=True,gridcolor='#34363D',zeroline=False,title=ylabel,linecolor='#B8BAC2',tickfont=dict(color='#E2E2E7')))
    return fig

def name_series(row,arq):
    return str(row['Modelo']) if row['Arquitectura']=='Tabular' else str(row['Arquitectura'])+' · '+str(row['Modelo'])

def create_lines(lines,metric,chosen=None,only=None,neon=True,first=0,window=12):
    fig=go.Figure()
    if lines.empty:return chart_style(fig,'Sin datos para estos filtros')
    visible=lines.copy()
    if only:visible=visible[visible['Dataset']==only]
    series=sorted(visible['Serie'].unique(),key=lambda s:(0 if s in MODELOS_BASE else 1,s))
    order=lines[['Dataset','Posicion_X']].drop_duplicates().sort_values('Posicion_X')
    labels=order['Dataset'].tolist() if not only else [only]
    for i,serie in enumerate(series):
        g=visible[visible['Serie']==serie].sort_values('Posicion_X')
        xx=g['Dataset'].astype(str)
        selected=(chosen==serie)
        col=COLORS[i%len(COLORS)]
        if neon:
            fig.add_trace(go.Scatter(x=xx,y=g[metric],mode='lines',line=dict(color=col,width=8 if selected else 5),opacity=.12 if not chosen or selected else .018,showlegend=False,hoverinfo='skip',legendgroup=serie))
        # use plotly event-selected points to update detailed dataset
        custom=g[['Dataset','Dataset_Real','Arquitectura','Modelo']].fillna('').astype(str).to_numpy()
        fig.add_trace(go.Scatter(x=xx,y=g[metric],mode='lines+markers',name=serie,
            line=dict(color=col,width=4 if selected else 2,dash='dash' if serie in MODELOS_BASE else 'solid'),
            marker=dict(size=10 if selected else 6,line=dict(color='#F2F2F2',width=1 if selected else 0)),
            opacity=1 if not chosen or selected else .25,customdata=custom,
            hovertemplate='<b>%{customdata[0]}</b><br>Modelo: '+serie+'<br>Valor: %{y:.5f}<extra></extra>',legendgroup=serie))
    chart_style(fig,ylabel=METRICAS.get(metric,metric),xlabel='Datasets ordenados')
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=labels,tickangle=-75,tickfont=dict(size=11,color='#FFFFFF'))
    if metric in MAYOR_MEJOR:fig.update_yaxes(range=[0,100],tickmode='linear',tick0=0,dtick=5,tickfont=dict(color='#FFFFFF'))
    else:fig.update_yaxes(tickfont=dict(color='#FFFFFF'))
    # Mantener tamaño adaptable a Streamlit. Un rango de 8-14 datasets
    # evita la saturacion sin recurrir a un iframe ancho e invisible.
    if not only and labels:
        first=min(max(0,int(first)),max(0,len(labels)-1))
        end=min(len(labels),first+int(window))
        fig.update_xaxes(range=[first-0.45,end-0.55])
    fig.update_layout(height=610,autosize=True,dragmode='pan',
        margin=dict(l=65,r=200,t=35,b=115),
        uirevision=f'{metric}-{only}-{len(labels)}')
    return fig

def jpg_graph(fig,filename):
    """Exporta JPG sin requerir Chrome/Kaleido; recrea lineas Plotly en Matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import to_rgba
    fig2,ax=plt.subplots(figsize=(17,8),dpi=150)
    fig2.patch.set_facecolor(BG);ax.set_facecolor(BG)
    for tr in fig.data:
        if tr.showlegend is False or tr.mode is None or 'lines' not in tr.mode:continue
        y=np.array(tr.y,dtype=float)
        x=list(tr.x)
        color=tr.line.color if tr.line and tr.line.color else '#00F5FF'
        ax.plot(x,y,color=color,linewidth=2,marker='o',markersize=3,label=tr.name)
    ax.tick_params(colors='white',axis='both',labelsize=9)
    for tick in ax.get_xticklabels():
        tick.set_rotation(75)
        tick.set_ha('right')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(alpha=.20,color='#FFFFFF');ax.set_ylabel(fig.layout.yaxis.title.text or '',color='white')
    if fig.layout.yaxis.range:ax.set_ylim(*fig.layout.yaxis.range)
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),facecolor=PANEL,labelcolor='white',fontsize=8)
    fig2.tight_layout()
    salida=io.BytesIO(); fig2.savefig(salida,facecolor=BG,bbox_inches='tight',format='jpg',dpi=160)
    plt.close(fig2);return salida.getvalue()

def jpg_table(data,filename,title='Tabla de resultados'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if data.empty:raise ValueError('No hay filas para exportar')
    frame=data.copy().head(65)
    frame=frame.iloc[:,:min(14,len(frame.columns))].copy()
    for c in frame.columns:
        if pd.api.types.is_float_dtype(frame[c]):frame[c]=frame[c].map(lambda v:f'{v:.4g}' if pd.notna(v) else '—')
    frame=frame.fillna('—').astype(str).map(lambda s:(s[:43]+'…') if len(s)>44 else s)
    w=max(12,min(26,1.8*len(frame.columns)))
    h=max(3,min(24,1.0+.37*(len(frame)+1)))
    f,ax=plt.subplots(figsize=(w,h),dpi=140);f.patch.set_facecolor(BG);ax.set_facecolor(BG);ax.axis('off')
    ax.set_title(title,loc='left',color='white',fontweight='bold',pad=14)
    tbl=ax.table(cellText=frame.values,colLabels=frame.columns,loc='upper center',cellLoc='left')
    tbl.auto_set_font_size(False);tbl.set_fontsize(7.5);tbl.scale(1,1.4)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor('#304562');cell.set_facecolor('#1C3155' if r==0 else (PANEL if r%2 else '#17253B'))
        cell.get_text().set_color('#FFFFFF')
    f.tight_layout();salida=io.BytesIO();f.savefig(salida,facecolor=BG,bbox_inches='tight',format='jpg',dpi=170)
    plt.close(f);return salida.getvalue()

def df_show(df,cols=None):
    if cols:df=df[[c for c in cols if c in df.columns]]
    st.dataframe(df,use_container_width=True,hide_index=True,height=min(490,90+len(df)*34))

def write_download(datos, nombre):
    st.download_button('Descargar JPG',data=datos,file_name=nombre,mime='image/jpeg',key='descargar_'+nombre)

st.title('TINTOlib  /  Análisis experimental')
st.caption('Resultados tabulares y representaciones visuales · Exploración y diagnóstico')
st.markdown('**Cargar resultados experimentales**')
st.caption('Explora los datos de ejemplo incluidos o carga tu propia carpeta de resultados CSV.')
modo_datos = st.radio(
    'Origen de los datos',
    ['Datos de ejemplo (automaticos)', 'Cargar mis resultados (carpeta CSV)'],
    horizontal=True,
    key='modo_datos'
)

if modo_datos == 'Datos de ejemplo (automaticos)':
    if not DEMO_ZIP.is_file():
        st.error('No se encontro demo/Local_ejemplo.zip en el repositorio. Agrega el ZIP de ejemplo a GitHub.')
        st.stop()
    try:
        df, meta, folds, dup, valid, fuentes, cantidad_archivos, tamanio_mb = cargar_ejemplo_zip(
            str(DEMO_ZIP), DEMO_ZIP.stat().st_mtime_ns
        )
    except Exception as exc:
        st.error(f'No se pudieron cargar los datos de ejemplo: {exc}')
        st.stop()
    st.info('Modo demostracion: conjunto de resultados de ejemplo. Esta carga no representa necesariamente los experimentos definitivos.')
    st.caption(f'Datos de ejemplo · Local · {cantidad_archivos} CSV · {tamanio_mb:.2f} MB')
else:
    st.caption('Selecciona Local o Datos_Analisis_Interactivo. Se necesitan Maestros/Resultados_Globales_Por_Modelo.csv y Datasets/Tabla_Datasets.csv.')
    archivos_usuario = st.file_uploader(
        'Carpeta de resultados CSV', type=['csv'], accept_multiple_files='directory',
        help='Si contiene Local y Colab se prioriza Local. Los CSV se procesan en memoria.',
        key='carga_carpeta'
    )
    if not archivos_usuario:
        st.info('Selecciona tu carpeta para analizar sus CSV. Puedes volver al modo de ejemplo cuando quieras.')
        st.stop()
    try:
        tamanio = sum(f.size for f in archivos_usuario)
        if tamanio > MAX_CSV_BYTES:
            raise ValueError('La carpeta supera el limite de 80 MB en CSV.')
        archivos_analisis, grupo_elegido = preparar_archivos_subidos(archivos_usuario)
        df, meta, folds, dup, valid, fuentes = cargar(archivos_analisis)
    except Exception as exc:
        st.error(f'No se pudieron cargar los CSV: {exc}')
        st.stop()
    st.caption(f'Carpeta cargada · origen: {grupo_elegido} · {len(archivos_analisis)} archivos de analisis · {tamanio / 1024**2:.2f} MB de CSV seleccionados')

if df.empty:
    st.error('El archivo global no tiene registros.')
    st.stop()

issues=calidad(df,meta,folds,dup,valid,fuentes)
with st.expander(f'Calidad y trazabilidad · {len(issues)} observaciones',expanded=not dup.empty):
    for issue in issues:
        if str(issue).startswith('No se detectaron'):
            st.success(issue)
        else:
            st.warning(issue)
    st.caption('Los hallazgos son descriptivos; cinco folds no implican significancia estadistica. No inferimos causalidad con datos agregados.')

k1,k2,k3,k4=st.columns(4)
k1.metric('Datasets',meta['Dataset'].nunique())
k2.metric('Combinaciones disponibles',len(df))
k3.metric('Registros por fold',len(folds))
k4.metric('Folds duplicados',len(dup))

tabs=st.tabs(['Líneas','Ganadores','Anomalías','Relaciones','Costo-beneficio','Comparador y auditoría'])

with tabs[0]:
    chartcol, controls=st.columns([4.6,1.25],gap='small')
    with controls, st.container(key='panel_explorar'):
        st.subheader('Explorar')
        metric=st.selectbox('Métrica',list(METRICAS),format_func=lambda x:METRICAS[x],key='line_metric')
        arq=st.selectbox('Arquitectura',['ResNet18','EfficientNet-B0','ViT','Todas','Tabular'],key='line_arch')
        criterio=st.selectbox('Ordenar datasets por',['Desbalance','Instancias','Features','Num_Clases','Categoria'],key='line_order')
        available=ordenar(meta,criterio)['Dataset'].astype(str).tolist()
        focus=st.selectbox('Dataset destacado',['Todos']+available,key='focus_ds')
        neons=st.toggle('Resaltar curvas',value=False)
        mode=st.radio('Mostrar',['Todos los datasets','Solo dataset destacado'],horizontal=False)
        if mode=='Solo dataset destacado' and focus=='Todos':st.caption('Selecciona un dataset para mostrar un solo punto.')
        sub_selection=st.selectbox('Resaltar una línea',['Ninguna']+MODELOS_BASE+[a+' · '+m for a in ARQUITECTURAS for m in METODOS])
        st.caption('Rendimiento en %; tiempos en segundos. Sin datos inventados.')
    lines=serie_lineas(df,meta,arq,metric,criterio)
    with chartcol:
        st.caption('Curvas por dataset. Navega por bloques con los controles inferiores o arrastra la grafica horizontalmente.')
        nav_a,nav_b=st.columns([1,2])
        with nav_b:
            viewport=st.select_slider('Datasets visibles',options=[8,10,12,14,16,20,24],value=12,key='line_window')
        maxstart=max(0,len(available)-viewport)
        if 'line_first' not in st.session_state:st.session_state.line_first=0
        if st.session_state.line_first>maxstart:st.session_state.line_first=maxstart
        with nav_a:
            first=st.number_input('Desde dataset n.º',min_value=1,max_value=maxstart+1,
                value=min(st.session_state.line_first+1,maxstart+1),step=1,key='line_start_display')-1
        st.session_state.line_first=int(first)
    if lines.empty:
        with chartcol:st.info('No existen resultados para esta selección.')
    else:
        series=lines['Serie'].unique().tolist()
        chosen=sub_selection if sub_selection in series else None
        only=focus if mode=='Solo dataset destacado' and focus!='Todos' else None
        fig=create_lines(lines,metric,chosen,only,neons,first,viewport)
        with chartcol:
            sel=st.plotly_chart(fig,use_container_width=True,key='main_chart',on_select='rerun',selection_mode='points')
            st.caption('Arrastra para desplazarte por el eje X; selecciona un punto para inspeccionarlo. Los rangos de Y avanzan de 5 en 5 cuando se trata de porcentajes.')
            if st.button('Preparar gráfica JPG',key='save_line'):
                try:write_download(jpg_graph(fig,f'Lineas_{arq}_{metric}_{criterio}_{datetime.now():%Y%m%d_%H%M%S}.jpg'),f'Lineas_{arq}_{metric}_{criterio}_{datetime.now():%Y%m%d_%H%M%S}.jpg')
                except Exception as e:st.error(f'No se pudo guardar la imagen: {e}')
        points=sel.selection.points if sel is not None and hasattr(sel,'selection') else []
        clicked=None
        if points:
            p=points[0]
            if isinstance(p,dict):
                if p.get('x') is not None:clicked=str(p['x'])
            else:clicked=str(getattr(p,'x',''))
        active=clicked if clicked in available else (focus if focus!='Todos' else available[0])
        st.subheader(f'Ficha de diagnóstico · {active}')
        dsdata=meta[meta['Dataset']==active]
        a,b=st.columns([1,2.7])
        with a:
            st.markdown('**Características del dataset**')
            if not dsdata.empty:
                info=dsdata.iloc[0]
                details=pd.DataFrame([{'Propiedad':c,'Valor':info.get(c,np.nan)} for c in ['Dataset','Dataset_Real','Instancias','Features','Num_Clases','Categoria','Desbalance','Metadata_Registro_Coincide'] if c in dsdata.columns])
                df_show(details)
                if st.button('Preparar características JPG'):
                    try:write_download(jpg_table(details,f'Metadatos_{active}_{datetime.now():%Y%m%d_%H%M%S}.jpg',f'Dataset {active}'),f'Metadatos_{active}_{datetime.now():%Y%m%d_%H%M%S}.jpg')
                    except Exception as e:st.error(str(e))
        with b:
            st.markdown('**Resultados completos: modelos tabulares y Deep Learning**')
            selected_data=df[df['Dataset']==active].copy()
            showcols=['Arquitectura','Modelo','Accuracy','F1_Score','ROC_AUC','Tiempo_s','Tiempo_Train_s','Tiempo_Eval_s','RAM_GB','VRAM_GB','Accuracy_STD','F1_Score_STD','ROC_AUC_STD','Accuracy_N_Validos']
            df_show(selected_data,showcols)
            if st.button('Preparar tabla JPG'):
                try:write_download(jpg_table(selected_data[[c for c in showcols if c in selected_data]],f'Resultados_{active}_{datetime.now():%Y%m%d_%H%M%S}.jpg',f'Resultados {active}'),f'Resultados_{active}_{datetime.now():%Y%m%d_%H%M%S}.jpg')
                except Exception as e:st.error(str(e))
            if st.button('Descargar resultados CSV'):
                st.download_button('Descargar CSV', selected_data.to_csv(index=False).encode('utf-8-sig'), file_name=f'Resultados_{active}.csv', mime='text/csv')
        with st.expander('Folds del dataset seleccionado'):
            if not folds.empty and 'Dataset' in folds:df_show(folds[folds['Dataset']==active],[ 'Arquitectura','Modelo','Fold','Accuracy','F1_Score','ROC_AUC','Tiempo_s','Fuente_CSV'])
            else:st.info('No hay folds individuales exportados.')

with tabs[1]:
    st.subheader('Victorias por métrica')
    st.caption('Se compara el mejor resultado de cada arquitectura DL con el mejor tabular, para cada dataset.')
    m=st.selectbox('Métrica de comparación',list(METRICAS),format_func=lambda v:METRICAS[v],key='win_metric')
    wins=tabla_victorias(df,m)
    if wins.empty:
        st.info('No hay pares tabular–Deep Learning completos para esta métrica.')
    else:
        counts=wins.groupby(['Arquitectura_DL','Ganador']).size().rename('Cantidad').reset_index()
        architectures=[a for a in ARQUITECTURAS if a in counts['Arquitectura_DL'].unique()]
        fig=go.Figure()
        for winner,color in [('Tabular','#E6E6EA'),('Deep Learning','#76C7C4')]:
            by_arch=counts[counts['Ganador']==winner].set_index('Arquitectura_DL')['Cantidad']
            fig.add_trace(go.Bar(name=winner,x=architectures,y=[int(by_arch.get(a,0)) for a in architectures],marker_color=color))
        fig.update_layout(barmode='group')
        chart_style(fig,'Victorias: mejor tabular vs mejor método visual',ylabel='Número de datasets',xlabel='Arquitectura comparada')
        fig.update_layout(height=440,margin=dict(l=55,r=170,t=60,b=85),
            title_font=dict(color='#FFFFFF'))
        st.plotly_chart(fig,use_container_width=True)
        emp=wins[wins['Ganador']=='Empate']
        if not emp.empty:
            st.caption(f'Empates exactos: {len(emp)} comparaciones (se conservan, pero no cuentan como victoria para ninguno de los enfoques).')
            with st.expander('Consultar empates'):
                df_show(emp,['Dataset','Arquitectura_DL','Modelo_DL','Modelo_Tabular',m+'_DL',m+'_Tabular'])
        st.markdown('**Modelos tabulares que obtuvieron el mejor resultado en su grupo**')
        tab=wins.groupby('Modelo_Tabular').agg(Comparaciones=('Ganador','size'),Victorias_Tabular=('Ganador',lambda x:int((x=='Tabular').sum())),Victorias_DL=('Ganador',lambda x:int((x=='Deep Learning').sum())),Empates=('Ganador',lambda x:int((x=='Empate').sum()))).reset_index()
        # No contar las tres comparaciones con una arquitectura como tres datasets independientes.
        mejores_tab=seleccionar_mejores(df[df['Arquitectura']=='Tabular'],m)
        tab_ds=mejores_tab['Modelo'].value_counts().rename_axis('Modelo_Tabular').reset_index(name='Datasets_mejor_tabular')
        tab=tab_ds.merge(tab,on='Modelo_Tabular',how='outer').fillna(0)
        df_show(tab)
        fig_tab=go.Figure(go.Bar(x=tab['Modelo_Tabular'],y=tab['Datasets_mejor_tabular'],
             name='Datasets ganados dentro del grupo tabular',marker_color='#A7A9AF',
             text=tab['Datasets_mejor_tabular'],textposition='outside'))
        chart_style(fig_tab,'Victorias internas de los tres clasificadores tabulares',
            'Clasificador tabular','Datasets donde fue el mejor tabular',False)
        fig_tab.update_layout(height=370,margin=dict(l=55,r=45,t=60,b=90),
            title_font=dict(color='#FFFFFF'))
        st.plotly_chart(fig_tab,use_container_width=True)
        st.caption('Este segundo gráfico muestra exclusivamente qué clasificador tabular fue el mejor dentro de su grupo. No representa victorias sobre Deep Learning.')
        st.markdown('**Ganador absoluto por dataset (3 tabulares + 3 arquitecturas visuales)**')
        absolutos=seleccionar_mejores(df,m,grupos=('Dataset',))
        absolutos=absolutos[['Dataset','Arquitectura','Modelo',m]].sort_values('Dataset')
        df_show(absolutos)
        st.markdown('**Comparación detallada entre enfoques**')
        df_show(wins)
        st.download_button('Descargar comparaciones CSV',wins.to_csv(index=False).encode('utf-8-sig'),'Comparaciones_Victorias.csv','text/csv')
        st.caption('Las victorias se determinan de manera descriptiva, no mediante pruebas de significancia. Cada arquitectura DL se compara contra el mejor tabular disponible.')

with tabs[2]:
    st.subheader('Anomalías y casos destacados')
    aa,bb,cc=st.columns(3)
    with aa:am=st.selectbox('Métrica',list(METRICAS),format_func=lambda v:METRICAS[v],key='alert_metric')
    with bb:threshold=st.number_input('Brecha mínima entre métodos',min_value=0.0,max_value=1e10,value=10.0,step=1.0)
    with cc:var=st.number_input('Desviación estándar mínima entre folds',min_value=0.0,value=10.0,step=1.0)
    al=alertas(df,am,threshold,var)
    st.metric('Alertas detectadas',len(al))
    if al.empty:
        st.info('Ningún caso supera los umbrales para esta métrica.')
    else:
        st.markdown('**Casos prioritarios y campo que destacó**')
        destaque=al.head(12).copy()
        destaque.insert(1,'Campo_analizado',METRICAS.get(am,am))
        destaque=destaque.rename(columns={'Tipo':'Motivo_de_alerta','Magnitud':'Diferencia_o_desviacion','Detalles':'Modelos_involucrados'})
        df_show(destaque,['Dataset','Campo_analizado','Motivo_de_alerta','Diferencia_o_desviacion','Unidad','Modelos_involucrados'])
        st.markdown('**Todas las anomalías detectadas**')
        df_show(al)
        st.download_button('Descargar anomalías CSV',al.to_csv(index=False).encode('utf-8-sig'),'Anomalias.csv','text/csv')
        d=st.selectbox('Inspeccionar dataset destacado',al['Dataset'].unique().tolist())
        detalle=al[al['Dataset']==d]
        st.markdown('**Campos y motivos de detección en el dataset seleccionado**')
        df_show(detalle,['Tipo','Magnitud','Unidad','Detalles','Arquitectura','Modelo'])
        info=meta[meta['Dataset']==d]
        if not info.empty:df_show(info,['Dataset','Dataset_Real','Instancias','Features','Num_Clases','Desbalance','Categoria'])
        df_show(df[df['Dataset']==d],['Arquitectura','Modelo',am,am+'_STD',am+'_N_Validos','Tiempo_s'])
    st.caption('Las anomalías describen diferencias y variabilidad. No determinan por sí mismas la causa ni la significancia estadística.')

with tabs[3]:
    st.subheader('Relaciones entre características y rendimiento')
    c1,c2,c3=st.columns(3)
    with c1:x=st.selectbox('Variable X',['Instancias','Features','Num_Clases','Desbalance','Tiempo_s','Tiempo_Train_s','Tiempo_Eval_s'])
    with c2:y=st.selectbox('Variable Y',list(METRICAS),format_func=lambda v:METRICAS[v],key='rel_metric')
    with c3:group=st.selectbox('Arquitectura',['Todas']+ARQUITECTURAS+['Tabular'],key='rel_arch')
    rd=df if group=='Todas' else df[df['Arquitectura']==group]
    rd=rd.dropna(subset=[x,y]).copy()
    if rd.empty:st.info('No hay observaciones completas para esta relación.')
    else:
        rd['Etiqueta']=rd['Arquitectura']+' · '+rd['Modelo']
        fig=px.scatter(rd,x=x,y=y,color='Arquitectura',hover_data=['Dataset','Modelo','Dataset_Real'],opacity=.82,render_mode='svg')
        chart_style(fig,'Relación exploratoria',xlabel=x,ylabel=METRICAS.get(y,y))
        fig.update_layout(margin=dict(l=55,r=220,t=45,b=85),legend=dict(x=1.02,y=1,xanchor='left',yanchor='top',orientation='v'))
        st.plotly_chart(fig,use_container_width=True)
        # No tratar cientos de filas por metodo/dataset como observaciones independientes del mismo dataset.
        st.caption('Cada punto es una combinación dataset–modelo, no un dataset independiente. Las correlaciones se calculan sobre un valor agregado por dataset para evitar seudorreplicación.')
        per_dataset=rd.groupby('Dataset',as_index=False)[[x,y]].mean()
        if x!=y and len(per_dataset)>=3:
            rho=per_dataset[[x,y]].corr(method='spearman').iloc[0,1]
            st.metric('Spearman exploratorio (agregado por dataset)',f'{rho:.3f}' if pd.notna(rho) else 'N/D')
        df_show(rd,['Dataset','Arquitectura','Modelo',x,y])

with tabs[4]:
    st.subheader('Eficiencia y frontera de Pareto')
    c1,c2,c3=st.columns(3)
    with c1:pm=st.selectbox('Rendimiento',list(MAYOR_MEJOR),index=0,key='pareto_metric')
    with c2:pt=st.selectbox('Costo temporal',['Tiempo_s','Tiempo_Train_s','Tiempo_Eval_s','Tiempo_Transformacion_s'])
    with c3:scope=st.selectbox('Alcance',['Promedio por modelo','Dataset específico'])
    if scope=='Dataset específico':
        pdataset=st.selectbox('Dataset',meta['Dataset'].tolist(),key='pareto_ds');p=df[df['Dataset']==pdataset].copy()
    else:
        p=df.groupby(['Arquitectura','Modelo'],as_index=False)[[pm,pt]].mean()
    p=frontera_pareto(p,pm,pt)
    if p.empty:st.info('No hay pares válidos de tiempo y rendimiento.')
    else:
        p['Etiqueta']=p['Arquitectura'].astype(str)+' · '+p['Modelo'].astype(str)
        fig=px.scatter(p,x=pt,y=pm,color='Pareto',hover_name='Etiqueta',color_discrete_map={True:'#00F5FF',False:'#7B61FF'},size_max=15)
        chart_style(fig,'Frontera de Pareto: más rendimiento con menos tiempo',xlabel=pt,ylabel=METRICAS.get(pm,pm))
        if (p[pt]>0).all() and p[pt].max()/p[pt].min()>20:fig.update_xaxes(type='log')
        st.plotly_chart(fig,use_container_width=True)
        df_show(p[['Arquitectura','Modelo',pt,pm,'Pareto']].sort_values(['Pareto',pm],ascending=[False,False]))
        st.caption('El tiempo comparado reproduce lo exportado. Puede excluir generación de imágenes: no se equipara automáticamente el costo total del pipeline DL con el tabular.')

with tabs[5]:
    st.subheader('Comparador entre dos métodos y auditoría de resultados')
    opts=sorted(df[['Arquitectura','Modelo']].drop_duplicates().itertuples(index=False,name=None))
    opts=[f'{a} · {m}' for a,m in opts]
    c1,c2,c3=st.columns(3)
    with c1:op1=st.selectbox('Modelo A',opts,index=0)
    with c2:op2=st.selectbox('Modelo B',opts,index=min(1,len(opts)-1))
    with c3:cm=st.selectbox('Métrica',list(METRICAS),format_func=lambda v:METRICAS[v],key='compare_metric')
    ar1,mo1=op1.split(' · ',1);ar2,mo2=op2.split(' · ',1)
    a=df[(df['Arquitectura']==ar1)&(df['Modelo']==mo1)][['Dataset',cm]].rename(columns={cm:'Modelo_A'})
    b=df[(df['Arquitectura']==ar2)&(df['Modelo']==mo2)][['Dataset',cm]].rename(columns={cm:'Modelo_B'})
    joined=a.merge(b,on='Dataset').dropna()
    if not joined.empty:
        joined['Diferencia_A_menos_B']=joined['Modelo_A']-joined['Modelo_B']
        fig=go.Figure()
        fig.add_trace(go.Bar(x=joined['Dataset'],y=joined['Diferencia_A_menos_B'],marker_color=['#00F5FF' if n>=0 else '#FF4DFF' for n in joined['Diferencia_A_menos_B']]))
        chart_style(fig,'Diferencia A − B por dataset','Dataset','Diferencia en unidades de la métrica',False)
        st.plotly_chart(fig,use_container_width=True)
        df_show(joined)
    else:st.info('Ambos métodos no tienen datasets con datos comparables.')
    with st.expander('Archivos de auditoría y origen',expanded=False):
        st.markdown('**Duplicados de folds**');df_show(dup)
        st.markdown('**Validación resumen oficial vs folds**');df_show(valid)
        st.markdown('**Fuentes identificadas**');df_show(fuentes)

st.divider()
st.caption('Interpretación científica: comparación descriptiva. No inferir causalidad, fuga de datos, significancia estadística ni diferencias de hardware sin pruebas adicionales. Los archivos CSV y JPG se descargan directamente desde el navegador.')
