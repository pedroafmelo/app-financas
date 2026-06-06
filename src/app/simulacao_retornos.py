# -*- coding: UTF-8 -*-
""" Import Modules """
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts
from backtest import Backtest, FACTOR_RENAME
from utils import get_config, sanitize_df
from calculating_factors import RiskFactors
from screening_ativos import _load_cotacoes_api, update_data
import time
from modelagem_retornos import EconometricPredict
from dateutil.relativedelta import relativedelta
from graphs import returns_forecast_chart, volatility_forecast_chart

# aqui teremos uma aba com um screening de ativos

config = get_config()
bt = Backtest()
risk_factors = RiskFactors()
predictor = EconometricPredict()

mean_opt = {
    "Constante": 'Constant', 
    "Zero": 'Zero', "Mínimos Quadrados": 'LS'
}
vol_opt = ["ARCH", "GARCH", "EGARCH", "GJR-GARCH"]
dist_opt = {"Normal": 'normal', "Gaussiana": 'gaussian', 
            "T": 't', "T-Student": 'studentst'}


@st.cache_data(show_spinner=False)
def get_dados_cad():

    dados_cadastrais = sanitize_df(
        pd.read_csv(config["screening_ativos"], encoding="latin")
        .set_axis(["Ticker", "Empresa", "Classe", "Subsetor", "Tipo", "Bolsa"], axis="columns")
        .drop(["Tipo", "Bolsa"], axis=1)
        ,
        cols_to_cast_float=[],
        cols_to_cast_int=[],
        col_to_cast_date= [],
        ticker_col="Ticker"
    ).drop("empresa", axis=1).assign(
        ticker_empresa = lambda x: x.apply(lambda row: row["Empresa"] + " - " + row["Ticker"], axis=1)
    )

    return dados_cadastrais


@st.cache_data(show_spinner=False)
def get_weighted_ret(tickers: list[str], log: bool = True) -> pd.Series:

    cotacoes_liq = _load_cotacoes_api()
    cotacoes_liq = cotacoes_liq[cotacoes_liq["ticker"].isin(tickers)]

    data_inicial = datetime.today() - relativedelta(years=predictor.modelling_time)
    retornos = (
        cotacoes_liq
        .sort_values(["ticker", "data"])
        .query("data >= @data_inicial")
        .assign(ret_arit=lambda df: df.groupby("ticker")["close"]
                                      .pct_change())
        .dropna(subset=["ret_arit"])
    )

    retorno_carteira = (
        retornos
        .groupby("data")["ret_arit"]
        .mean()
    )

    if log:
        retorno_carteira = np.log1p(retorno_carteira)

    retorno_carteira.name = "retorno_carteira"
    
    return retorno_carteira * 100


def render_simualtions():


    st.header("Simulação de Retornos")
    st.markdown(
        """
        <h5 style='color: grey; text-align: left; margin-bottom: 4px'>Faça a simulação dos retornos da sua carteira 📈</h5>
        <hr style='border-top: 1px solid #f5770c; margin-top: 4px; margin-bottom: 0'>

        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='margin-top: 10px;'>&nbsp;</p>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns([6, 3])

    with st.sidebar:

        ult_att = pd.to_datetime(pd.read_csv(
        f"{config["data_path"]}/filtros_risco.csv"
            )["ult_atualizacao"].values[-1])

        st.write(f"Dados atualizados em {datetime.strftime(ult_att, format="%d/%m/%Y")}")

        if st.button("Atualizar Dados"):
            update_data()

    # parametros dos modelos

    c1, c2 = st.columns([6, 4])

    c2.markdown("<h4>Selecione os Hiperparâmetros do modelo</h4>", unsafe_allow_html=True)

    with c2.container():

        st.html("<span class='any_container'></span>")
        vol = st.selectbox("Selecione a função da volatilidade", vol_opt)
        mean = st.selectbox("Selecione a função da média", list(mean_opt.keys()))
        vol = "GARCH" if vol == "GJR-GARCH" else vol
        dist = st.selectbox("Selecione a distribuição dos retornos", list(dist_opt.keys()))
        # if mean == ""
        p = st.slider("Selecione o Valor do parâmetro P", 1, 15, step=1)
        o = st.slider("Selecione o Valor do parâmetro O", 1, 15, step=1)
        q = st.slider("Selecione o Valor do parâmetro Q", 1, 15, step=1)

    with c1:
        col1, col2, col3 = st.columns([4, 0.1, 3])
        cadastrais = get_dados_cad()

        indiv_carteira = col1.toggle("Selecionar ativos da carteira")
        
        select_box_placeholder = col1.empty()

        if indiv_carteira:
            select_box_placeholder.empty()
            col3.space("small")

            if "carteira" not in st.session_state:
                warn = st.empty()
                with warn:
                    st.warning("Você ainda não tem ativos em sua carteira... selecione apenas um ou crie a carteira na aba de Screening!")
                time.sleep(5)
                warn.empty()

            else:
                ativos = st.session_state["carteira"]
        
        else:
            with select_box_placeholder:
                ativos = st.selectbox("Selecione o ativo para simular os retornos", 
                            cadastrais["ticker_empresa"], 
                            cadastrais["Ticker"].to_list().index("PETR4")
                            )[-5:]
            
        
        otimizar = col3.toggle("Otimizar modelo (Escolha Vol)")
        col3.space("small")
        if col3.button("Realizar simulação dos retornos"):

            retornos = get_weighted_ret([ativos], log=True)
            ticker_label = ativos if isinstance(ativos, str) else ", ".join(ativos)

            if otimizar:
                with st.spinner("Realizando a otimização do seu modelo..."):
                    metrics, best_params, pred = predictor.optimize_model(retornos, vol)
                    model = predictor.model(retornos)
                    pred_ret, pred_var = pred.mean, pred.variance
            else:
                with st.spinner("Ajustando o modelo e gerando previsões..."):
                    model = predictor.model(
                        retornos, mean_opt[mean], vol, p, o, q,
                        dist_opt[dist]
                    )
                    pred_ret, pred_var = predictor.predict(model, method="simulation",
                                            horizon=5, simulations=100)

            if pred_ret is not None and not (isinstance(pred_ret, list) and len(pred_ret) == 0):
                st.session_state["sim_retornos"] = retornos
                st.session_state["sim_pred_ret"] = pred_ret
                st.session_state["sim_pred_var"] = pred_var
                st.session_state["sim_ticker_label"] = ticker_label
            else:
                st.error("Não foi possível gerar as previsões. Verifique os parâmetros do modelo.")

        if "sim_pred_ret" in st.session_state:
            st.markdown("### Retornos Históricos e Previstos")
            st_echarts(
                options=returns_forecast_chart(
                    st.session_state["sim_retornos"],
                    st.session_state["sim_pred_ret"],
                    title=st.session_state["sim_ticker_label"],
                ),
                height=420,
            )

    if "sim_pred_var" in st.session_state:
        st.markdown("---")
        st.markdown("### Volatilidade Histórica e Prevista")
        st_echarts(
            options=volatility_forecast_chart(
                st.session_state["sim_retornos"],
                st.session_state["sim_pred_var"],
                title=st.session_state["sim_ticker_label"],
            ),
            height=420,
        )




if __name__ == "__main__":

    render_simualtions()