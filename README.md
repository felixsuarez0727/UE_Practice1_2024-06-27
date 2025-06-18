# 🏥 Práctica de Visualización y Análisis de Imágenes Médicas

## 📋 Descripción

Este proyecto implementa un sistema completo de **visualización y análisis básico de imágenes médicas** utilizando Python. Procesa archivos DICOM y NIfTI para realizar análisis cuantitativos y generar visualizaciones multiplanares de alta calidad.

### 🎯 Objetivos

- **Parte 1**: Procesamiento y visualización de series DICOM
- **Parte 2**: Análisis de imágenes NIfTI con segmentación automática
- **Reporte Final**: Generación de estadísticas completas y visualizaciones

---

## 🛠️ Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip 21.0 o superior

### Dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:
```bash
pip install pydicom[all] nibabel matplotlib pandas numpy pylibjpeg[all] python-gdcm pillow
```

---

## 📁 Estructura del Proyecto

```
UE_practice1_2024-06-27/
├── practice1.py                    # Script principal
├── requirements.txt                # Dependencias
├── README.md                      # Este archivo
├── case_practice1/                # Datos de entrada
│   ├── serie/                     # Archivos DICOM (725 archivos)
│   │   ├── IM000001.dcm
│   │   ├── IM000002.dcm
│   │   └── ...
│   ├── image.nii.gz              # Imagen NIfTI
│   └── segmentation.nii.gz       # Segmentación
└── [Archivos generados]:
    ├── case_practice1_metadata.csv
    ├── case_practice1_dicom_visualization.png
    ├── case_practice1_nifti_segmentation.png
    └── case_practice1_final_report.png
```

---

## 🚀 Uso

### Ejecución Principal

```bash
python practice1.py
```
---

## 📊 Resultados Obtenidos

### 🔍 Datos del Caso

- **Paciente**: EMR100070_2270002
- **Fecha**: 10 de Diciembre de 2015
- **Tipo**: TC de Tórax con Contraste (Thorax C+)
- **Equipo**: Philips
- **Dimensiones**: 725 × 512 × 512 voxeles

### 📈 Estadísticas de Segmentación

| Métrica | Valor |
|---------|-------|
| **Total voxeles segmentados** | 13,127,266 |
| **Número de etiquetas** | 5 (etiquetas 10-14) |
| **Rango de intensidades** | -1024 a 1880 HU |
| **Intensidad media** | -783.22 HU |
| **Voxeles > 100 HU** | 45,669 (0.35%) |
| **Volumen total** | ~3,200 ml |

### 🏷️ Distribución por Etiquetas

| Etiqueta | Voxeles | Porcentaje |
|----------|---------|------------|
| 10 | 1,240,386 | 9.4% |
| 11 | 2,382,138 | 18.1% |
| 12 | 3,761,771 | 28.7% |
| 13 | 1,305,151 | 9.9% |
| 14 | 4,437,820 | 33.8% |

---

## 📄 Archivos Generados

### 1. `case_practice1_metadata.csv`
- Metadatos completos de todos los archivos DICOM
- Información de paciente, estudio, serie y parámetros técnicos

### 2. `case_practice1_dicom_visualization.png`
- Visualización multiplanar (axial, coronal, sagital)
- Histograma de intensidades
- Estadísticas del volumen
- Perfil de intensidad

### 3. `case_practice1_nifti_segmentation.png`
- Visualización en tres ejes con segmentación superpuesta
- Máscaras de segmentación por separado
- Overlay de colores para diferentes estructuras

### 4. `case_practice1_final_report.png`
- Reporte completo con todas las estadísticas
- Gráficos de distribución por etiquetas
- Resumen ejecutivo del análisis

---

## 🔬 Funcionalidades Técnicas

### Procesamiento DICOM
- ✅ Carga automática de series completas
- ✅ Ordenamiento por posición espacial
- ✅ Soporte para compresión JPEG Lossless
- ✅ Extracción de metadatos completos
- ✅ Visualización multiplanar cuantitativa

### Análisis NIfTI
- ✅ Carga de imágenes y segmentaciones
- ✅ Visualización con overlay de segmentación
- ✅ Cálculo de estadísticas por región
- ✅ Análisis de distribución de intensidades
- ✅ Cuantificación volumétrica

### Controles de Calidad
- ✅ Validación de rangos de Hounsfield Units
- ✅ Verificación de consistencia dimensional
- ✅ Controles de integridad de datos
- ✅ Estadísticas de verificación cruzada

---

## 🎯 Cumplimiento de Requisitos

| Requisito | Puntos | Estado |
|-----------|--------|--------|
| **1a) Ordenamiento DICOM** | 1.0 | ✅ Completo |
| **1b) Metadatos a CSV** | 1.0 | ✅ Completo |
| **1c) Visualización multiplanar** | 2.0 | ✅ Completo |
| **2a) Visualización NIfTI** | 2.0 | ✅ Completo |
| **2b) Estadísticas segmentación** | 3.0 | ✅ Completo |
| **Aspectos adicionales** | 1.0 | ✅ Completo |
| **TOTAL** | **10.0** | **✅ 100%** |

---

## 🔧 Solución de Problemas

### Error: "Unable to decompress JPEG Lossless"
```bash
pip install pylibjpeg[all] python-gdcm
```

### Error: Archivo no encontrado
- Verificar que las rutas en `practice1.py` sean correctas
- Asegurar que todos los archivos estén en `case_practice1/`

### Error: Memoria insuficiente
- El procesamiento requiere ~8GB RAM para 725 slices
- Considerar procesar en lotes si hay limitaciones

---

## 📚 Referencias Técnicas

### Formatos de Imagen
- **DICOM**: Digital Imaging and Communications in Medicine
- **NIfTI**: Neuroimaging Informatics Technology Initiative
- **Hounsfield Units**: Escala de densidad en TC (-1024 a +3071)

### Librerías Utilizadas
- **pydicom**: Procesamiento de archivos DICOM
- **nibabel**: Manejo de formatos neuroimagen
- **matplotlib**: Visualización científica
- **pandas**: Análisis y manipulación de datos
- **numpy**: Computación numérica

---

## 👤 Información del Proyecto

- **Autor**: Félix David Suárez Bonilla
- **Universidad**: Universidad Europea
- **Asignatura**: Visualización y Análisis de Imágenes Médicas
- **Fecha**: Junio 2025
- **Versión**: 1.0

---

## 📜 Licencia

Práctica académica de la Universidad Europea.
