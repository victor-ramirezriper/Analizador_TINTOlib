# Analizador TINTOlib — Streamlit Community Cloud

Aplicación de análisis de resultados experimentales TINTOlib. Conserva los paneles de **Líneas, Ganadores, Anomalías, Relaciones, Costo-beneficio y Comparador y auditoría**.

## Carga de resultados desde el navegador

1. Ejecuta el exportador de datos del proyecto para obtener la carpeta `Datos_Analisis_Interactivo`.
2. Abre la aplicación y selecciona **Carpeta de resultados CSV**.
3. Selecciona una carpeta que contenga la estructura siguiente. También puedes seleccionar directamente `Local`:

```
Datos_Analisis_Interactivo/
└── Local/
    ├── Maestros/
    │   ├── Resultados_Globales_Por_Modelo.csv       [obligatorio]
    │   └── Resultados_Por_Fold.csv                  [opcional]
    ├── Datasets/
    │   └── Tabla_Datasets.csv                      [obligatorio]
    └── Auditoria/
        ├── Duplicados_Folds.csv                    [opcional]
        ├── Validacion_Resumen_vs_Folds.csv         [opcional]
        └── Auditoria_Fuentes.csv                   [opcional]
```

Si se selecciona una carpeta con `Local` y `Colab` al mismo tiempo, se usa **Local**. No se combinan registros de ambos orígenes. Se rechazan rutas duplicadas dentro del conjunto elegido. No es necesario subir las imágenes originales, resultados por época ni los directorios del entrenamiento: solo los CSV producidos por el exportador. El límite interno es 80 MB en total, contando los CSV elegidos.

**Importante:** el analizador no trabaja directamente con los `Baseline_*_FOLDS.csv` ni con los directorios de modelos sin procesar; primero hay que generar los CSV maestros mediante el exportador. Los resultados de un único fold/holdout son válidos y no se marcan como incompletos únicamente por contener un fold. Su desviación estándar entre folds puede ser `NaN`.

Los archivos se leen en memoria en la sesión del navegador y **no se incluyen en el repositorio GitHub**. Evita subir datasets confidenciales a un servicio público sin autorización. Las descargas JPG/CSV se realizan desde la interfaz y no se guardan de forma permanente en el servidor.

## Ejecutar localmente

Necesitas Python 3.12 o una versión compatible.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Crea un repositorio en GitHub y sube los archivos de esta carpeta al **directorio raíz** del repositorio: `app.py`, `analisis_core.py`, `requirements.txt`, `.streamlit/config.toml`, `README.md` y `.gitignore`.
2. Entra en https://share.streamlit.io y conecta tu cuenta de GitHub.
3. Selecciona el repositorio y la rama, e indica **Main file path**: `app.py`.
4. En la configuración avanzada, selecciona **Python 3.12** (si está disponible) y despliega la aplicación.
5. Comparte la URL pública `.streamlit.app`. Los visitantes podrán subir sus propias carpetas desde el navegador.

La carpeta `.streamlit` debe subirse, aunque empieza con punto. No subas datos experimentales ni credenciales al repositorio.

## Dependencias

Consulta `requirements.txt`; la interfaz utiliza `streamlit>=1.55` porque la carga de carpetas requiere `accept_multiple_files="directory"`.

## Alcance de la validación

Se verificaron las funciones de lectura, reconocimiento de la estructura de carpetas, selección de `Local`, detección de archivos faltantes/duplicados, análisis descriptivo y compatibilidad del caso holdout con un juego de CSV sintéticos. La ejecución final en Streamlit Community Cloud requiere publicar el repositorio y verificar los CSV reales exportados.
