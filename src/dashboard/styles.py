import streamlit as st

def apply_styles():
    st.markdown("""
    <style>

    header {
        visibility: hidden;
    }

    [data-testid="stAppViewContainer"] {
        padding-top: 0rem;
    }

    /* Force Plotly titles */
    .js-plotly-plot .plotly .gtitle {
        fill: #ffffff !important;
    }

    /* Axis titles */
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle {
        fill: #ffffff !important;
    }

    /* Legend text */
    .js-plotly-plot .plotly .legend text {
        fill: #ffffff !important;
    }

    /* Tick labels */
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #e6edf6 !important;
    }

    html, body, [data-testid="stAppViewContainer"]{
        background:#0a1222;
        color:#ffffff !important;
        font-family: Inter, sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"]{
        background:#071124;
        border-right:1px solid #1b2b4a;
        color:#ffffff !important;
    }

    /* Force ALL text */
    * {
        color:#e6edf6 !important;
    }

    /* Headings brighter */
    h1, h2, h3 {
        color:#ffffff !important;
    }

    /* Panels (your cards.py) */
    .panel-card{
        background:#0f1b33;
        border:1px solid #1f335a;
        padding:16px;
        border-radius:10px;
        color:#e6edf6 !important;
    }

    .section-title{
        font-size:18px;
        font-weight:600;
        margin-bottom:10px;
        color:#ffffff !important;
    }

    /* Metrics */
    [data-testid="metric-container"]{
        background:#0f1b33;
        border:1px solid #1f335a;
        padding:18px;
        border-radius:10px;
    }

    /* Buttons */
    .stButton > button{
        color:#ffffff !important;
    }

    /* Inputs */
    input, textarea{
        color:#ffffff !important;
        background-color:#0f1b33 !important;
    }

    /* Labels (very important) */
    label {
        color:#e6edf6 !important;
    }

    /* Plotly container */
    .stPlotlyChart{
        background:#0f1b33;
        border:1px solid #1f335a;
        border-radius:10px;
        padding:10px;
    }

    footer{visibility:hidden;}
    #MainMenu{visibility:hidden;}

    </style>
    """, unsafe_allow_html=True)