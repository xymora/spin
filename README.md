# spin

Aplicación interactiva desarrollada con Streamlit para visualizar, filtrar y analizar clientes bancarios segmentados, con integración en tiempo real a Firestore.

## 🚀 Funcionalidades

- Visualización del dataset `segmentation_data_recruitment.csv`
- Filtros dinámicos por múltiples columnas categóricas
- Integración con Firebase Firestore
- Notebook de carga automática desde CSV a Firestore
- Interfaz amigable desarrollada con Streamlit

## 📁 Estructura del Proyecto

```
spin/
├── notebooks/
│   ├── segmentation_dashboard.ipynb
│   └── segmentation_data_recruitment.csv
├── app/
│   └── streamlit_app.py
├── .streamlit/
│   └── secrets.toml
├── firebase_key.json
├── requirements.txt
└── README.md
```

## ⚙️ Requisitos

- Python 3.8 o superior
- Librerías:
  - Streamlit
  - Firebase Admin
  - Pandas

## ▶️ Ejecución local

```bash
streamlit run app/streamlit_app.py
```

## 📌 Autor

**Álvaro Rodrigo Moctezuma Ramírez**, Data Scientist especializado en desarrollo de dashboards interactivos, segmentación de clientes y automatización con Firebase y Python.
