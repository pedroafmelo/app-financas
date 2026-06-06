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
import os

# aqui teremos uma aba com um screening de ativos

config = get_config()
bt = Backtest()
risk_factors = RiskFactors()

MODELOS = {
    "capm": 1,
    "3 fatores": 3,
    "5 fatores": 5,
    "completo": len(FACTOR_RENAME.values())
}

MODELOS_DETALHES = {
    "CAPM": {
        "key": "capm",
        "fatores": ["Mkt-Rf"],
    },
    "3 Fatores": {
        "key": "3 fatores",
        "fatores": ["Mkt-Rf", "SMB", "HML"],
    },
    "5 Fatores": {
        "key": "5 fatores",
        "fatores": ["Mkt-Rf", "SMB", "HML", "WML", "IML"],
    },
    "Completo (6 fatores)": {
        "key": "completo",
        "fatores": ["Mkt-Rf", "SMB", "HML", "WML", "IML", "BAB"],
    },
}

@st.cache_data(show_spinner=False)
def get_expected_returns(tickers: list[str], 
                         model, monthly_factors,
                         portfolio, rfr):
    
    
    filtered_factors = ["ano_mes"] + list(FACTOR_RENAME.values())[:MODELOS[model]]
    monthly_factors = monthly_factors[filtered_factors]
    
    betas = bt.get_factor_betas(tickers, window_months=24, 
                                monthly_factors=monthly_factors, 
                                portfolio=portfolio, rfr=rfr)
    
    filtered_betas = {fac: beta for fac, beta in betas.items() if fac in filtered_factors}
    
    return filtered_betas

@st.cache_data(show_spinner=False)
def get_screening_df():
    
    ano_atual = datetime.today().year
    tickers = (
        pd.read_csv(config["elegibilidade_final_file_path"])
        .query("ano == @ano_atual")
        [["ticker"]]
    )

    cotacoes = sanitize_df(
            bt.get_data(config["cotacoes_liquidez_data_feed"], feed=True)
            .set_axis(["ticker", "data", "close", "qt_negs", "vol_negociado"], axis="columns"),
            cols_to_cast_float=["close"],
            cols_to_cast_int=[],
            col_to_cast_date="data",
        )[["ticker", "data", "close"]].dropna(subset="close")

    monthly_factors = bt.get_monthly_factor_returns()

    rfr = (
            bt.get_rfr_series("monthly")
            .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
            )

    betas_tickers = {model: [] for model in MODELOS.keys()}
    expected_returns = {model: [] for model in MODELOS.keys()}
    useful_tickers = []
    for tick in tickers["ticker"].to_list():
        portfolio = bt.get_portfolio_monthly_returns([tick], cotacoes[cotacoes["ticker"] == tick])
        for model in MODELOS.keys():
            try:
                betas = get_expected_returns(
                    [tick], model, monthly_factors,
                    portfolio, rfr
                    )
                exp_return = bt.get_expected_return(
                    betas, window_months=24, rfr_monthly=rfr
                )
                betas_tickers[model].append(betas)
                expected_returns[model].append(exp_return)
                if tick not in useful_tickers:
                    useful_tickers.append(tick)
        
            except ValueError:
                continue

    df_betas = pd.DataFrame()

    for model, betas in betas_tickers.items():

        df_temp_betas = pd.DataFrame(betas, index=useful_tickers)
        df_temp_betas.columns = [f"{col}_{model}" if col in FACTOR_RENAME.values() else col for col in df_temp_betas.columns]
        df_betas = pd.concat([df_betas, df_temp_betas], axis=1)

    df_exp = pd.DataFrame()

    for model, exp in expected_returns.items():

        df_temp_exp = pd.DataFrame(exp, index=useful_tickers)
        df_temp_exp.columns = [f"ret_esp_{model}"]
        df_exp = pd.concat([df_exp, df_temp_exp], axis=1)

    df_risk = (
        pd.merge(df_betas, df_exp, right_index=True,
                 left_index=True,
                 how="inner").reset_index().rename(columns={0: "ticker", 
                                                            None: "ticker",
                                                            "index": "ticker"})
        .assign(
            ult_atualizacao = cotacoes["data"].max()
        )
    )

    df_risk.to_csv(f"{config["data_path"]}/filtros_risco.csv", index=False)


@st.cache_data(show_spinner=False)
def _load_cotacoes_api():
    return sanitize_df(
        bt.get_data("cotacoes_liquidez_ativos")
        .set_axis(["ticker", "data", "close", "qt_negs", "vol_negociado"], axis="columns"),
        cols_to_cast_float=["close"],
        cols_to_cast_int=[],
        col_to_cast_date="data",
    )[["ticker", "data", "close"]].dropna(subset="close")

@st.cache_data(show_spinner=False)
def update_data():
    df_cotacoes = sanitize_df(
                bt.get_data(config["cotacoes_liquidez_data_feed"], feed=True)
                .set_axis(["ticker", "data", "close", "qt_negs", "vol_negociado"], axis="columns"),
                cols_to_cast_float=["close"],
                cols_to_cast_int=[],
                col_to_cast_date="data",
            )
    
    df_elegibilidade = (
                bt.get_data(config["elegibilidade_data_feed"], feed=True)
            )

    df_cotacoes.to_csv(f"{config["data_path"]}/cotacoes_liquidez_ativos.csv")
    df_elegibilidade.to_csv(f"{config["data_path"]}/elegibilidade_ativos.csv")

    get_screening_df()
    risk_factors.transform_trimestral_indicators()


@st.cache_data(show_spinner=False)
def _compute_betas_alfa(ticker: str, model_key: str):
    cotacoes = _load_cotacoes_api()
    tick_cotacoes = cotacoes[cotacoes["ticker"] == ticker]
    portfolio = bt.get_portfolio_monthly_returns([ticker], tick_cotacoes)
    monthly_factors = bt.get_monthly_factor_returns()
    rfr = bt.get_rfr_series("monthly").assign(ano_mes=lambda x: x["data"].dt.to_period("M"))

    n_factors = MODELOS[model_key]
    factor_names = list(FACTOR_RENAME.values())[:n_factors]
    filtered_factors = monthly_factors[["ano_mes"] + factor_names]

    betas = bt.get_factor_betas([ticker], window_months=24,
                                monthly_factors=filtered_factors,
                                portfolio=portfolio, rfr=rfr)
    exp_return = bt.get_expected_return(betas, window_months=24, rfr_monthly=rfr,
                                        monthly_factors=filtered_factors)
    return betas, exp_return


@st.cache_data(show_spinner=False)
def _compute_attribution(ticker: str, model_key: str, start_date: str, end_date: str):
    """Calcula atribuição de performance sem chamar a API de cotações duas vezes."""
    cotacoes = _load_cotacoes_api()
    tick_cotacoes = cotacoes[cotacoes["ticker"] == ticker]
    portfolio = bt.get_portfolio_monthly_returns([ticker], tick_cotacoes)
    monthly_factors = bt.get_monthly_factor_returns()
    rfr = bt.get_rfr_series("monthly")
    rfr_period = rfr.assign(ano_mes=lambda x: x["data"].dt.to_period("M"))

    n_factors = MODELOS[model_key]
    factor_names = list(FACTOR_RENAME.values())[:n_factors]
    all_factor_cols = list(FACTOR_RENAME.values())
    filtered_factors = monthly_factors[["ano_mes"] + factor_names]

    betas = bt.get_factor_betas([ticker], window_months=24,
                                monthly_factors=filtered_factors,
                                portfolio=portfolio, rfr=rfr_period)

    sd, ed = pd.to_datetime(start_date), pd.to_datetime(end_date)

    mf = monthly_factors.assign(data=lambda x: x["ano_mes"].dt.to_timestamp())
    period_factors = mf[(mf["data"] >= sd) & (mf["data"] <= ed)]
    factor_cumret = (1 + period_factors[all_factor_cols]).prod() - 1

    port = portfolio.assign(data=lambda x: x["ano_mes"].dt.to_timestamp())
    period_port = port[(port["data"] >= sd) & (port["data"] <= ed)]
    portfolio_cumret = float((1 + period_port["retorno_carteira"]).prod() - 1)

    rfr_d = rfr.assign(data=lambda x: pd.to_datetime(x["data"]))
    period_rfr = rfr_d[(rfr_d["data"] >= sd) & (rfr_d["data"] <= ed)]
    rfr_cumret = float((1 + period_rfr["cdi"]).prod() - 1)

    attribution = {"CDI (Rf)": rfr_cumret}
    total_factor = 0.0
    for f in factor_names:
        contrib = betas.get(f, 0.0) * float(factor_cumret.get(f, 0.0))
        attribution[f] = contrib
        total_factor += contrib
    attribution["Alfa"] = portfolio_cumret - rfr_cumret - total_factor
    attribution["Retorno Real"] = portfolio_cumret
    return attribution


def _echarts_line(labels, values, title, y_format="{value}", color="#f5770c"):
    safe_vals = [
        round(float(v), 4)
        if v is not None and not (isinstance(v, float) and np.isnan(v))
        else None
        for v in values
    ]
    area_color = "rgba(245,119,12,0.06)" if color == "#f5770c" else "rgba(76,175,80,0.06)"
    return {
        "backgroundColor": "#0E1117",
        "title": {"text": title, "textStyle": {"color": "#FFFFFF", "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": "#AAAAAA", "rotate": 45},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#AAAAAA", "formatter": y_format},
        },
        "dataZoom": [{"show": True, "start": 0, "end": 100}, {"type": "inside"}],
        "series": [{
            "type": "line",
            "data": safe_vals,
            "lineStyle": {"color": color, "width": 2},
            "itemStyle": {"color": color},
            "symbol": "circle",
            "symbolSize": 4,
            "connectNulls": False,
            "areaStyle": {"color": area_color},
        }],
    }


def _fund_attribution_chart(attribution: dict) -> dict:
    labels = list(attribution.keys())
    values = [round(v * 100, 3) for v in attribution.values()]
    bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in values]
    return {
        "backgroundColor": "#0E1117",
        "title": {"text": "Decomposição dos Retornos por Fator",
                  "textStyle": {"color": "#FFFFFF"}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                    "formatter": "{b}: {c}%"},
        "xAxis": {
            "type": "value",
            "axisLabel": {"color": "#AAAAAA", "formatter": "{value}%"},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "yAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": "#FFFFFF"},
        },
        "series": [{
            "type": "bar",
            "data": [{"value": v, "itemStyle": {"color": c}} for v, c in zip(values, bar_colors)],
            "label": {"show": True, "formatter": "{c}%", "color": "#FFFFFF", "position": "right"},
        }],
    }


def fundamentals_details():

    dados_cadastrais = (
        sanitize_df(
            pd.read_csv(config["screening_ativos"], encoding="latin")
            .set_axis(["ticker", "Empresa", "Classe", "Subsetor", "Tipo", "Bolsa"], axis="columns")
            .drop(["Tipo", "Bolsa"], axis=1),
            cols_to_cast_float=[],
            cols_to_cast_int=[],
            col_to_cast_date=[],
        )
        .drop("empresa", axis=1)
        .assign(
            ticker_empresa=lambda x: x.apply(
                lambda row: row["Empresa"] + f" - {row['ticker']}", axis=1
            )
        )
    )

    indicadores_fund = (
        pd.read_csv(config["indicadores_trimestrais_file_path"])
        .assign(roe=lambda x: x["lucro"] / x["patrimonio_liquido"])
        .replace([np.inf, -np.inf], np.nan)
    )

    st.markdown("<h4>Fundamentos</h4>", unsafe_allow_html=True)
    st.markdown(
        "<h5 style='color: grey; margin-top: 0'>Detalhamento de fundamentos e análise quantitativa de um ativo</h5>",
        unsafe_allow_html=True,
    )

    st.space("medium")

    acao = st.selectbox(
        "Selecione o ativo para detalhar",
        dados_cadastrais["ticker_empresa"],
        key="selectbox_fund_detail", index = dados_cadastrais["ticker"].to_list().index("PETR4")
    )
    ticker = acao.split(" - ")[-1]

    df_ativo = (
        indicadores_fund[indicadores_fund["ticker"] == ticker]
        .sort_values(["data", "tri"])
        .reset_index(drop=True)
    )

    if df_ativo.empty:
        st.warning("Sem dados fundamentalistas para o ativo selecionado.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────
    with st.spinner("Carregando dados de cotação..."):
        cotacoes_all = _load_cotacoes_api()
    cotacoes_ticker = cotacoes_all[cotacoes_all["ticker"] == ticker].sort_values("data")

    preco_atual = float(cotacoes_ticker["close"].iloc[-1]) if not cotacoes_ticker.empty else None

    ret_12m = None
    if not cotacoes_ticker.empty:
        monthly = (
            cotacoes_ticker
            .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
            .groupby("ano_mes", as_index=False)
            .last()
            .sort_values("ano_mes")
        )
        if len(monthly) >= 13:
            preco_ant = monthly["close"].iloc[-13]
            preco_now = monthly["close"].iloc[-1]
            if preco_ant and preco_ant > 0:
                ret_12m = (preco_now / preco_ant) - 1

    ult_pl = df_ativo[df_ativo["preco_lucro"] > 0].tail(1)
    pl_atual = float(ult_pl["preco_lucro"].values[0]) if not ult_pl.empty else None

    ult_roe = df_ativo[df_ativo["roe"].notna()].tail(1)
    roe_atual = float(ult_roe["roe"].values[0]) if not ult_roe.empty else None

    st.markdown("##### KPIs Atuais")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Preço Atual", f"R$ {preco_atual:.2f}" if preco_atual is not None else "N/D")
    k2.metric(
        "Rentabilidade (12m)",
        f"{ret_12m * 100:.1f}%" if ret_12m is not None else "N/D",
    )
    k3.metric("P/L Atual", f"{pl_atual:.1f}x" if pl_atual is not None else "N/D")
    k4.metric("ROE Atual", f"{roe_atual * 100:.1f}%" if roe_atual is not None else "N/D")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Histórico ROE e P/L ───────────────────────────────────────────────
    labels = df_ativo["tri"].tolist()
    roe_vals = df_ativo["roe"].tolist()
    pl_vals = df_ativo["preco_lucro"].replace(0, np.nan).tolist()

    col_roe, col_pl = st.columns(2)
    with col_roe.container():
        st.html("<span class='any_container'></span>")
        st_echarts(options=_echarts_line(labels, roe_vals, "Histórico ROE"), height=300)
    with col_pl.container():
        st.html("<span class='any_container'></span>")
        st_echarts(
            options=_echarts_line(
                labels, pl_vals, "Histórico P/L", y_format="{value}x", color="#4CAF50"
            ),
            height=300,
        )

    # ── Balanço Patrimonial ───────────────────────────────────────────────
    st.markdown("##### Balanço Patrimonial (Histórico Trimestral)")
    tabela_bp = (
        df_ativo[["tri", "lucro", "patrimonio_liquido", "fin_cp", "fin_lp", "eqv_caixa"]]
        .rename(columns={
            "tri": "Trimestre",
            "lucro": "Lucro Líquido",
            "patrimonio_liquido": "Patrimônio Líquido",
            "fin_cp": "Endiv. Curto Prazo",
            "fin_lp": "Endiv. Longo Prazo",
            "eqv_caixa": "Equiv. de Caixa",
        })
        .iloc[::-1]
        .reset_index(drop=True)
    )
    num_cols_bp = [
        "Lucro Líquido", "Patrimônio Líquido",
        "Endiv. Curto Prazo", "Endiv. Longo Prazo", "Equiv. de Caixa",
    ]
    st.dataframe(
        tabela_bp,
        hide_index=True,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="R$ %,.0f")
            for col in num_cols_bp
        },
    )

    # ── Detalhamento Quantitativo ─────────────────────────────────────────
    st.divider()
    st.markdown("<h4>Detalhamento Quantitativo</h4>", unsafe_allow_html=True)

    modelo_selecionado = st.selectbox(
        "Modelo de precificação",
        options=list(MODELOS_DETALHES.keys()),
        key="modelo_quant_detail",
    )
    model_key = MODELOS_DETALHES[modelo_selecionado]["key"]
    fatores = MODELOS_DETALHES[modelo_selecionado]["fatores"]

    with st.spinner("Estimando betas e retorno esperado..."):
        try:
            betas, exp_return = _compute_betas_alfa(ticker, model_key)
        except ValueError as e:
            st.error(f"Não foi possível calcular betas: {e}")
            return

    alpha_annual = float((1 + betas.get("alpha", 0.0)) ** 12 - 1)

    st.markdown(f"##### Betas, Alfa e Retorno Esperado — {modelo_selecionado}")
    metric_cols = st.columns(len(fatores) + 2)
    metric_cols[0].metric(
        "Ret. Esperado (a.a.)",
        f"{exp_return * 100:.2f}%",
        help="E(R) = (1 + Rf + Σ βᵢ·prêmioᵢ)¹² − 1",
    )
    metric_cols[1].metric(
        "Alfa (a.a.)",
        f"{alpha_annual * 100:.2f}%",
        help="Alfa estimado por OLS (janela 24 meses), anualizado.",
    )
    for i, fator in enumerate(fatores):
        metric_cols[i + 2].metric(f"β {fator}", f"{betas.get(fator, 0.0):.3f}")

    # ── Decomposição de Retornos ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Decomposição dos Retornos")

    if cotacoes_ticker.empty:
        st.info("Sem dados de cotação para calcular a decomposição.")
        return

    monthly_dates = (
        cotacoes_ticker
        .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
        .groupby("ano_mes", as_index=False)
        .last()
        .sort_values("ano_mes")
    )
    min_date = monthly_dates["ano_mes"].dt.to_timestamp().min().date()
    max_date = monthly_dates["ano_mes"].dt.to_timestamp().max().date()
    default_start = max_date - timedelta(days=365)
    if default_start < min_date:
        default_start = min_date

    dc1, dc2 = st.columns(2)
    attr_start = dc1.date_input(
        "Data de início", value=default_start,
        min_value=min_date, max_value=max_date,
        key="attr_start_fund",
    )
    attr_end = dc2.date_input(
        "Data de fim", value=max_date,
        min_value=min_date, max_value=max_date,
        key="attr_end_fund",
    )

    if attr_start >= attr_end:
        st.warning("A data de início deve ser anterior à data de fim.")
        return

    with st.spinner("Calculando decomposição de retornos..."):
        try:
            attribution = _compute_attribution(ticker, model_key, str(attr_start), str(attr_end))
        except ValueError as e:
            st.error(str(e))
            return

    real_ret = attribution.pop("Retorno Real")
    st.metric("Retorno Real no Período", f"{real_ret * 100:.2f}%")

    with st.container():
        st.html("<span class='any_container'></span>")
        st_echarts(
            options=_fund_attribution_chart(attribution),
            height=max(300, len(attribution) * 55),
        )
    attr_df = pd.DataFrame([
        {"Componente": k, "Contribuição (%)": round(v * 100, 3)}
        for k, v in attribution.items()
    ])
    with st.expander("Ver tabela de decomposição"):
        st.dataframe(attr_df, hide_index=True)



def render_resumo_geral():


    st.header("Screening de Ativos")
    st.markdown(
        """
        <h5 style='color: grey; text-align: left; margin-bottom: 4px'>Filtro de risco de ativos brasileiros 🇧🇷</h5>
        <hr style='border-top: 1px solid #f5770c; margin-top: 4px; margin-bottom: 0'>

        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='margin-top: 10px;'>&nbsp;</p>
        """, unsafe_allow_html=True)
    
    st.markdown("""<h4>Filtre os ativos da B3 pelo seu risco e retorno esperado</h4>""", unsafe_allow_html=True)

    c1, c2 = st.columns([6, 3])

    with st.sidebar:

        ult_att = pd.to_datetime(pd.read_csv(
        f"{config["data_path"]}/filtros_risco.csv"
            )["ult_atualizacao"].values[-1])

        st.write(f"Dados atualizados em {datetime.strftime(ult_att, format="%d/%m/%Y")}")

        if st.button("Atualizar Dados"):
            update_data()

    dados_cadastrais = sanitize_df(
        pd.read_csv(config["screening_ativos"], encoding="latin")
        .set_axis(["Ticker", "Empresa", "Classe", "Subsetor", "Tipo", "Bolsa"], axis="columns")
        .drop(["Tipo", "Bolsa"], axis=1)
        ,
        cols_to_cast_float=[],
        cols_to_cast_int=[],
        col_to_cast_date= [],
        ticker_col="Ticker"
    ).drop("empresa", axis=1)
    
    risk_df = pd.read_csv(
        f"{config["data_path"]}/filtros_risco.csv"
    )

    ret_esp_cols = [col for col in risk_df.columns if "ret_esp" in col]
    risk_cols = ["ticker", "Mkt-Rf_capm"]

    for col in ret_esp_cols:
        risk_df[col] = risk_df[col] * 100

    risk_df = risk_df[risk_cols + ret_esp_cols]
    risk_df.columns = ["Ticker", "Beta (CAPM)", "Ret. Esp (CAPM)",
                       "Ret. Esp (3 Fat.)", "Ret. Esp (5 Fat.)", "Ret. Esp (Completo)"]

    final_df = (
        dados_cadastrais.merge(risk_df, on="Ticker")
    )

    # c2.space("small")
    with c2.container():

        st.html("<span class='any_container'></span>")
        min_beta = st.slider("Beta mínimo", 
                             risk_df["Beta (CAPM)"].min(), risk_df["Beta (CAPM)"].max())
        min_ret_capm = st.slider("Retorno esperado mínimo (CAPM)", 
                                 risk_df["Ret. Esp (CAPM)"].min(), risk_df["Ret. Esp (CAPM)"].max())
        min_ret_3 = st.slider("Retorno esperado mínimo (3 Fat.)",
                                 risk_df["Ret. Esp (3 Fat.)"].min(), risk_df["Ret. Esp (3 Fat.)"].max())
        min_ret_5 = st.slider("Retorno esperado mínimo (5 Fat.)",
                                 risk_df["Ret. Esp (5 Fat.)"].min(), risk_df["Ret. Esp (5 Fat.)"].max())
        min_ret_comp = st.slider("Retorno esperado mínimo (Completo)",
                                 risk_df["Ret. Esp (Completo)"].min(), risk_df["Ret. Esp (Completo)"].max())
        
        set_excluidos = st.multiselect("Setores Excluídos", final_df["Subsetor"])


    with c1.container():
        st.html("<span class='any_container'></span>")

        final_df = (
            final_df[
                (~final_df["Subsetor"].isin(set_excluidos)) & 
                (final_df["Beta (CAPM)"] > min_beta) & 
                (final_df["Ret. Esp (CAPM)"] > min_ret_capm) & 
                (final_df["Ret. Esp (3 Fat.)"] > min_ret_3) & 
                (final_df["Ret. Esp (5 Fat.)"] > min_ret_5) & 
                (final_df["Ret. Esp (Completo)"] > min_ret_comp) 
            ]
        )
        st.dataframe(final_df, hide_index=True, height=500)

    
    add = st.button("Adicionar ativos à carteira simulada.")

    @st.dialog("Adicionar à carteira?")
    def confirm_portfolio(ativos):

        st.write("Já existem ativos na sua carteira... deseja adicionar ou substituir?")

        st.space("medium")
        c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
        adicionar = c2.button("Adicionar")
        substituir = c3.button("Substituir")

        if adicionar:
            st.session_state["carteira"].extend(ativos)
            st.rerun()
        elif substituir:
            st.session_state["carteira"] = ativos
            st.rerun()


    if add:
        acoes = final_df["Ticker"].to_list()
        if "carteira" not in st.session_state:
            st.session_state["carteira"] = acoes
        elif "carteira" in st.session_state:
            confirm_portfolio(acoes)

    st.divider()
    
    fundamentals_details()

if __name__ == "__main__":
    render_resumo_geral()