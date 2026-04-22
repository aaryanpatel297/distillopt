import streamlit as st

def panel(title):
    st.markdown(f'<div class="panel-card"><div class="section-title">{title}</div>', unsafe_allow_html=True)

def end_panel():
    st.markdown("</div>", unsafe_allow_html=True)