import os
import re
import gc
import torch
import logging
import tempfile
import whisperx
import streamlit as st

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="24 Horas | Transcriptor Turbo",
    page_icon="🔴",
    layout="wide"
)

# --- CONFIGURACIÓN DE TOKEN Y LOGS ---
mi_hf_token = ""  # Pega tu token de huggingface.co aquí para diarización
logging.getLogger("whisperx").setLevel(logging.ERROR)

# --- ESTILOS CORPORATIVOS TVN ---
rojo_tvn = "#E2001A"
gris_f = "#1E1E24"
gris_c = "#2D2D34"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {gris_f};
        color: white;
        font-family: 'Roboto', sans-serif;
    }}
    .header-prensa {{
        background-color: {rojo_tvn};
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        color: white !important;
        margin-bottom: 25px;
    }}
    .header-prensa h1 {{
        color: white !important;
        margin: 0;
        font-size: 28px;
    }}
    .header-prensa p {{
        margin: 5px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }}
    div[data-baseweb="textarea"] {{
        background-color: {gris_c} !important;
        border-left: 5px solid {rojo_tvn} !important;
    }}
    textarea {{
        color: white !important;
        font-size: 16px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DEL MODELO (Modificado para Streamlit Cloud) ---
@st.cache_resource
def cargar_modelos():
    # Forzamos CPU e int8 para que sea compatible con el entorno gratuito de Streamlit Cloud
    device = "cpu"
    compute_type = "int8"
    model_idx = whisperx.load_model("turbo", device, compute_type=compute_type, language="es")
    return model_idx, device

try:
    model, device = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar el modelo base: {e}")

# --- FUNCIONES DE OPTIMIZACIÓN ---
def quitar_bucles_redundantes(texto):
    palabras = texto.split()
    if not palabras: return ""
    i, resultado, n = 0, [], len(palabras)
    while i < n:
        bucle_detectado = False
        for k in range(1, 16):
            if i + 2 * k <= n:
                chunk1 = palabras[i : i + k]
                chunk2 = palabras[i + k : i + 2 * k]
                c1_norm = [re.sub(r'[^\w]', '', w.lower()) for w in chunk1]
                c2_norm = [re.sub(r'[^\w]', '', w.lower()) for w in chunk2]
                if c1_norm == c2_norm and "".join(c1_norm) != "":
                    veces = 0
                    while i + (veces + 2) * k <= n:
                        siguiente_chunk = palabras[i + (veces + 1) * k : i + (veces + 2) * k]
                        if [re.sub(r'[^\w]', '', w.lower()) for w in siguiente_chunk] == c1_norm: veces += 1
                        else: break
                    resultado.extend(chunk1)
