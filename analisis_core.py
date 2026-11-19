"""Nucleo de analisis reproducible. No modifica archivos del experimento."""
from pathlib import Path
from io import BytesIO
from typing import Mapping
import numpy as np
import pandas as pd

MODELOS_BASE = ['Random_Forest','SVM','MLP_Clasico']
METODOS = ['TINTO','IGTD','REFINED','BARGRAPH','DISTANCEMATRIX','COMBINATION','SUPERTML','FEATUREWRAP','BIE','FOTOMICS','DEEPINSIGHT']
ARQUITECTURAS = ['ResNet18','EfficientNet-B0','ViT']
METRICAS = {'Accuracy':'Accuracy (%)','F1_Score':'F1-macro (%)','ROC_AUC':'ROC-AUC (%)','Balanced_Accuracy':'Balanced Accuracy (%)','Precision':'Precision (%)','Recall':'Recall (%)','Tiempo_s':'Tiempo total (s)','Tiempo_Train_s':'Entrenamiento (s)','Tiempo_Eval_s':'Clasificacion (s)','Tiempo_Transformacion_s':'Transformacion (s)','RAM_GB':'RAM (GB)','VRAM_GB':'VRAM (GB)','GPU_Usage_%':'GPU (%)','CPU_Usage_%':'CPU (%)'}
MAYOR_MEJOR = {'Accuracy','F1_Score','ROC_AUC','Balanced_Accuracy','Precision','Recall'}
META = ['Dataset','Dataset_Real','Instancias','Features','Num_Clases','Categoria','Desbalance']

# Estructura generada por exportar_datos_graficas.py.
# La carga puede realizarse desde un directorio local (compatibilidad) o
# desde archivos CSV subidos en la sesion, sin escribir datos en el servidor.
ARCHIVOS_ANALISIS = {
    'Maestros/Resultados_Globales_Por_Modelo.csv',
    'Maestros/Resultados_Por_Fold.csv',
    'Datasets/Tabla_Datasets.csv',
    'Auditoria/Duplicados_Folds.csv',
    'Auditoria/Validacion_Resumen_vs_Folds.csv',
    'Auditoria/Auditoria_Fuentes.csv',
}
OBLIGATORIOS = {
    'Maestros/Resultados_Globales_Por_Modelo.csv',
    'Datasets/Tabla_Datasets.csv',
}


def _leer_csv(origen):
    if origen is None:
        return pd.DataFrame()
    if isinstance(origen, (bytes, bytearray)):
        if not origen:
            return pd.DataFrame()
        for codificacion in ('utf-8-sig', 'latin-1'):
            try:
                return pd.read_csv(BytesIO(origen), encoding=codificacion, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError('No se pudo decodificar uno de los CSV.')
    ruta = Path(origen)
    if not ruta.is_file():
        return pd.DataFrame()
    for codificacion in ('utf-8-sig', 'latin-1'):
        try:
            return pd.read_csv(ruta, encoding=codificacion, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'No se pudo decodificar {ruta.name}.')


def preparar_archivos_subidos(archivos):
    """Selecciona un unico juego de CSV segun su estructura relativa.

    Soporta seleccionar Local, Datos_Analisis_Interactivo o la carpeta que
    contiene cualquiera de ellas. Prefiere Local cuando hay ambos origenes.
    Nunca utiliza las rutas recibidas para acceder al sistema de archivos.
    """
    entradas = []
    for archivo in archivos:
        nombre = str(archivo.name).replace('\\', '/')
        partes = [p for p in nombre.split('/') if p and p != '.']
        if '..' in partes or not partes or not partes[-1].lower().endswith('.csv'):
            continue
        partes_lower = [p.casefold() for p in partes]
        for idx in range(len(partes) - 1):
            seccion = partes_lower[idx]
            if seccion not in ('maestros', 'datasets', 'auditoria'):
                continue
            if idx != len(partes) - 2:
                continue
            ruta_rel = f'{partes[idx]}/{partes[idx+1]}'
            nombre_can = next((x for x in ARCHIVOS_ANALISIS if x.casefold() == ruta_rel.casefold()), None)
            if nombre_can is not None:
                # Se preserva el origen (Local/Colab), sin mezclar carpetas.
                prefijo = partes_lower[:idx]
                if 'local' in prefijo:
                    grupo = 'local'
                elif 'colab' in prefijo:
                    grupo = 'colab'
                else:
                    grupo = 'directo'
                entradas.append((grupo, nombre_can, archivo))
            break

    grupos = {}
    for grupo, ruta, archivo in entradas:
        grupos.setdefault(grupo, {}).setdefault(ruta, []).append(archivo)
    candidatos = [g for g in ('local', 'directo', 'colab') if OBLIGATORIOS.issubset(grupos.get(g, {}))]
    if not candidatos:
        presentes = sorted({ruta for _,ruta,_ in entradas})
        faltan = sorted(OBLIGATORIOS - set(presentes))
        raise ValueError('Faltan los CSV obligatorios de la estructura exportada: ' + ', '.join(faltan) + '. Selecciona la carpeta que contiene Maestros, Datasets y Auditoria.')
    grupo = candidatos[0]
    ambiguos = [ruta for ruta, fs in grupos[grupo].items() if len(fs)>1]
    if ambiguos:
        raise ValueError('Se encontraron archivos duplicados con la misma ruta relativa: '+ ', '.join(sorted(ambiguos)) + '. Sube una sola carpeta exportada.')

    resultado = {ruta: fs[0].getvalue() for ruta,fs in grupos[grupo].items()}
    return resultado, grupo


def cargar(origen):
    """Carga CSV de un directorio o de un dict de archivos CSV en memoria."""
    if isinstance(origen, Mapping):
        def obtener(ruta): return origen.get(ruta)
    else:
        base = Path(origen)
        def obtener(ruta): return base.joinpath(*ruta.split('/'))
    maestro = 'Maestros/Resultados_Globales_Por_Modelo.csv'
    metadatos = 'Datasets/Tabla_Datasets.csv'
    if obtener(maestro) is None or (not isinstance(origen, Mapping) and not Path(obtener(maestro)).is_file()):
        raise FileNotFoundError('No se encuentra '+maestro+'. Ejecuta primero el exportador.')
    if obtener(metadatos) is None or (not isinstance(origen, Mapping) and not Path(obtener(metadatos)).is_file()):
        raise FileNotFoundError('No se encuentra '+metadatos+'. Ejecuta primero el exportador.')
    df = _leer_csv(obtener(maestro))
    md = _leer_csv(obtener(metadatos))
    columnas=['Dataset','Dataset_Real','Arquitectura','Modelo','Tipo']
    faltan=[c for c in columnas if c not in df.columns]
    if faltan: raise ValueError('Faltan columnas en Resultados_Globales_Por_Modelo.csv: '+str(faltan))
    if 'Dataset' not in md.columns:
        raise ValueError('Falta la columna Dataset en Tabla_Datasets.csv.')
    if md['Dataset'].duplicated().any():raise ValueError('Dataset duplicado en Tabla_Datasets.csv; revisar exportacion.')
    for col in METRICAS:
        if col in df.columns: df[col]=pd.to_numeric(df[col],errors='coerce')
    for col in ['Instancias','Features','Num_Clases','Desbalance']:
        for frame in (df,md):
            if col in frame.columns:frame[col]=pd.to_numeric(frame[col],errors='coerce')
    for col in ['N_Registros']+[f'{m}_N_Validos' for m in METRICAS]+[f'{m}_STD' for m in METRICAS]:
        if col in df.columns: df[col]=pd.to_numeric(df[col],errors='coerce')
    folds=_leer_csv(obtener('Maestros/Resultados_Por_Fold.csv'))
    if not folds.empty:
        for col in ['Fold']+list(METRICAS):
            if col in folds:folds[col]=pd.to_numeric(folds[col],errors='coerce')
    auditoria=_leer_csv(obtener('Auditoria/Duplicados_Folds.csv'))
    validacion=_leer_csv(obtener('Auditoria/Validacion_Resumen_vs_Folds.csv'))
    fuentes=_leer_csv(obtener('Auditoria/Auditoria_Fuentes.csv'))
    return df,md,folds,auditoria,validacion,fuentes

def ordenar(meta,col):
    if col not in meta.columns: return meta.sort_values('Dataset',kind='stable').reset_index(drop=True)
    return meta.sort_values([col,'Dataset'],na_position='last',kind='stable').reset_index(drop=True)

def serie_lineas(df,meta,arq,metrica,orden):
    grupos=['Tabular'] if arq=='Tabular' else [arq,'Tabular'] if arq!='Todas' else ARQUITECTURAS+['Tabular']
    sub=df[df['Arquitectura'].isin(grupos)].copy()
    sub=sub[sub['Modelo'].isin(MODELOS_BASE+METODOS)]
    if metrica not in sub: return pd.DataFrame()
    base=ordenar(meta,orden)
    cols=['Dataset','Dataset_Real','Instancias','Features','Num_Clases','Categoria','Desbalance']
    base=base[[c for c in cols if c in base]].copy()
    base['Posicion_X']=np.arange(1,len(base)+1)
    # La posicion X es determinada exclusivamente por metadatos, no por rendimiento.
    sub=sub.drop(columns=[c for c in cols if c!='Dataset' and c in sub],errors='ignore')
    result=sub.merge(base,on='Dataset',how='inner',validate='many_to_one')
    result['Serie']=np.where(result['Arquitectura']=='Tabular',result['Modelo'],result['Arquitectura']+' · '+result['Modelo'])
    return result.sort_values(['Posicion_X','Arquitectura','Modelo'],kind='stable')

def seleccionar_mejores(df,metrica,grupos=('Dataset','Arquitectura')):
    if metrica not in df:return pd.DataFrame()
    x=df.dropna(subset=[metrica]).copy()
    if x.empty:return x
    x=x.sort_values(metrica,ascending=metrica not in MAYOR_MEJOR,kind='stable')
    return x.drop_duplicates(list(grupos),keep='first')

def tabla_victorias(df,metrica):
    x=seleccionar_mejores(df,metrica)
    if x.empty:return pd.DataFrame()
    bas=x[x['Arquitectura']=='Tabular']
    dls=x[x['Arquitectura']!='Tabular']
    if bas.empty or dls.empty:return pd.DataFrame()
    out=dls.merge(bas,on='Dataset',suffixes=('_DL','_Tabular'))
    out['Diferencia_DL_menos_Tabular']=out[metrica+'_DL']-out[metrica+'_Tabular']
    out['Ventaja_DL']=out['Diferencia_DL_menos_Tabular'] if metrica in MAYOR_MEJOR else -out['Diferencia_DL_menos_Tabular']
    out['Ganador']=np.select([out['Ventaja_DL']>1e-9,out['Ventaja_DL']< -1e-9],['Deep Learning','Tabular'],default='Empate')
    keep=['Dataset','Dataset_Real_DL','Arquitectura_DL','Modelo_DL','Modelo_Tabular',metrica+'_DL',metrica+'_Tabular','Diferencia_DL_menos_Tabular','Ganador','Instancias_DL','Features_DL','Num_Clases_DL','Categoria_DL','Desbalance_DL']
    return out[[c for c in keep if c in out]].sort_values(['Dataset','Arquitectura_DL'])

def comparar_2_modelos(df,metrica,modelo_a,arq_a,modelo_b,arq_b):
    a=df[(df['Arquitectura']==arq_a)&(df['Modelo']==modelo_a)][['Dataset',metrica]].rename(columns={metrica:'A'})
    b=df[(df['Arquitectura']==arq_b)&(df['Modelo']==modelo_b)][['Dataset',metrica]].rename(columns={metrica:'B'})
    out=a.merge(b,on='Dataset').dropna(subset=['A','B'])
    out['Diferencia_A_menos_B']=out['A']-out['B']
    return out

def alertas(df,metrica,umbral=10,variabilidad=10):
    """Reglas descriptivas, no inferencia de causa ni significancia estadistica."""
    if metrica not in df:return pd.DataFrame()
    rows=[]
    asc=metrica not in MAYOR_MEJOR
    for ds,g in df.groupby('Dataset'):
        g=g.dropna(subset=[metrica]);
        if len(g)<2:continue
        high=g.sort_values(metrica,ascending=asc).iloc[0]
        low=g.sort_values(metrica,ascending=not asc).iloc[0]
        delta=float(abs(high[metrica]-low[metrica]))
        if delta>=umbral:
            rows.append({'Dataset':ds,'Tipo':'Brecha entre metodos','Magnitud':delta,'Unidad':'puntos porcentuales' if metrica in MAYOR_MEJOR else 'unidad de metrica','Detalles':f"Mejor: {high['Arquitectura']} / {high['Modelo']} ({high[metrica]:.3f}); peor: {low['Arquitectura']} / {low['Modelo']} ({low[metrica]:.3f})",'Arquitectura':str(high['Arquitectura']),'Modelo':str(high['Modelo'])})
    stdcol=metrica+'_STD'
    if stdcol in df:
        for _,r in df[pd.to_numeric(df[stdcol],errors='coerce')>=variabilidad].iterrows():
            n=r.get(metrica+'_N_Validos',np.nan)
            rows.append({'Dataset':r['Dataset'],'Tipo':'Alta dispersion entre folds','Magnitud':float(r[stdcol]),'Unidad':'desviacion estandar (misma unidad)','Detalles':f"{r['Arquitectura']} / {r['Modelo']}; n validos: {n}",'Arquitectura':r['Arquitectura'],'Modelo':r['Modelo']})
    return pd.DataFrame(rows).sort_values('Magnitud',ascending=False) if rows else pd.DataFrame(columns=['Dataset','Tipo','Magnitud','Unidad','Detalles','Arquitectura','Modelo'])

def frontera_pareto(frame,metrica='Accuracy',tiempo='Tiempo_s'):
    z=frame.dropna(subset=[metrica,tiempo]).copy()
    z=z[(z[tiempo]>=0)]; z['Pareto']=False
    high=metrica in MAYOR_MEJOR
    for i,r in z.iterrows():
        better=(z[tiempo]<=r[tiempo]) & ((z[metrica]>=r[metrica]) if high else (z[metrica]<=r[metrica]))
        strict=(z[tiempo]<r[tiempo]) | ((z[metrica]>r[metrica]) if high else (z[metrica]<r[metrica]))
        z.at[i,'Pareto']=not bool((better&strict).any())
    return z

def calidad(df,meta,folds,dup,valid,fuentes):
    issues=[]
    if not dup.empty:issues.append(f'Hay {len(dup)} registros involucrados en duplicados de fold. Las medias exportadas pueden incluirlos.')
    missing=meta[[c for c in ['Instancias','Features','Num_Clases','Desbalance'] if c in meta]].isna().sum()
    if missing.sum():issues.append('Hay metadatos faltantes: '+', '.join(f'{c}={n}' for c,n in missing.items() if n))
    if not valid.empty:
        dif=[c for c in valid if c.startswith('Diferencia_')]
        bad=pd.DataFrame({c:pd.to_numeric(valid[c],errors='coerce').abs()>1e-4 for c in dif}) if dif else pd.DataFrame()
        if not bad.empty and bad.any().any():issues.append(f'{int(bad.any(axis=1).sum())} registros con diferencias entre resumen tabular oficial y media de folds (tolerancia 0.0001 unidades de exportacion).')
    if not folds.empty and 'Fold' in folds:
        gr=folds.groupby(['Dataset','Arquitectura','Modelo'],dropna=False)['Fold'].nunique()
        less=gr[(gr<5) & (gr!=1)]
        if len(less):issues.append(f'{len(less)} combinaciones dataset/arquitectura/modelo tienen 2–4 folds identificados; pueden ser parciales. Los resultados de 1 fold se tratan como holdout.')
    if not issues:issues=['No se detectaron los problemas incluidos en estas comprobaciones automaticas; esto no garantiza ausencia de otros errores.']
    return issues
