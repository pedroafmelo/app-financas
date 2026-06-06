# -*- coding: UTF-8 -*-
""" Import Modules """
import streamlit as st
from datetime import datetime, date
from os import path
from pathlib import Path

# local imports
from utils import get_config, update_config

config = get_config()


def build_multi_page_app():

    """
    renders multi page app
    """

    st.set_page_config("Screening de Ativos", 
                    page_icon = "🪙", 
                    layout="wide")

    st.html(Path(__file__).parent / "static/index.html")
    

    # portfolios = st.Page("portfolios.py", title="Portfólios dos Fatores", icon=":material/wallet:")
    # methodology = st.Page("methodology.py", title="Metodologia dos Fatores", icon=":material/docs:", default=True)
    # factors = st.Page("factors.py", title="Análise Técnica de Fatores", icon=":material/target:")
    simulator = st.Page("simulacao_retornos.py", title="Simulador", icon=":material/thumb_up:")
    screening = st.Page("screening_ativos.py", title="Screening", icon=":material/filter_alt:")
    
    pg = st.navigation(
        [screening, simulator]
    )

    with st.sidebar:

        st.logo(config["logo_ufpb"], icon_image=config["logo_ufpb"], size="large")
        

    pg.run()


if __name__ == "__main__":
    build_multi_page_app()