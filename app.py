import streamlit as st

from src.dashboard.styles import apply_styles
from src.dashboard.predictor import predict
from src.dashboard.plots import *

apply_styles()
st.title("Distillation Column Optimizer")


st.set_page_config(
    page_title="Distillation Column Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:

    st.header("Column Controls")

    z = st.slider("Feed Composition",0.1,0.9,0.5)
    T = st.slider("Feed Temperature (°C)",60,130,90)
    R = st.slider("Reflux Ratio",1.2,6.0,2.5)
    P = st.slider("Pressure (kPa)",80,200,130)
    N = st.slider("Number of Trays",10,40,25)
    F = st.slider("Feed Flow (kmol/h)",50,500,200)

# Prediction
purity,bottoms,energy=predict(z,T,R,P,N,F)

# KPI cards
col1,col2,col3=st.columns(3)

col1.metric("Distillate Purity",f"{purity*100:.2f}%")
col2.metric("Bottoms Composition",f"{bottoms*100:.2f}%")
col3.metric("Energy Consumption",f"{energy/1000:.2f} MW")

st.markdown("---")
st.subheader("Performance Gauges")

# Gauges + sensitivity
col1,col2,col3=st.columns([1,1,1.8])

with col1:
    st.plotly_chart(
        purity_gauge(purity*100),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        energy_gauge(energy/1000),
        use_container_width=True
    )

with col3:
    st.plotly_chart(
        reflux_sensitivity(z,T,R,P,N,F),
        use_container_width=True
    )

st.markdown("---")
st.subheader("Operating Envelope")

col4,col5=st.columns([1.4,1])

with col4:
    st.plotly_chart(
        purity_contour(z,T,R,P,N,F),
        use_container_width=True
    )

with col5:
    st.plotly_chart(
    mccabe_thiele(z, purity, bottoms),
    use_container_width=True
)

st.markdown("---")
st.subheader("Sensitivity Analysis")

col6,col7,col8=st.columns(3)

with col6:
    st.plotly_chart(
        tray_sensitivity(z,T,R,P,N,F),
        use_container_width=True
    )

with col7:
    st.plotly_chart(
        pressure_sensitivity(z,T,R,P,N,F),
        use_container_width=True
    )

with col8:
    st.plotly_chart(
        operating_radar(purity,energy,N,F,R),
        use_container_width=True
    )