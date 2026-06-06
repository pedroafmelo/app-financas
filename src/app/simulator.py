# -*- coding: UTF-8 -*-
"""Simulador de Carteira — Factor Analyzer."""

import streamlit as st
from functools import reduce
from datetime import datetime, date, timedelta
from os import path
from pathlib import Path
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import statsmodels.api as sm

from utils import get_config, sanitize_df
from backtest import Backtest, FACTOR_RENAME

config = get_config()

FACTOR_DISPLAY = ["Mkt-Rf", "SMB", "HML", "WML", "IML", "BAB"]

# ------------------------------------------------------------------ #
#  Data preparation                                                    #
# ------------------------------------------------------------------ #

market_cap_data_feed = path.join(config["config_vars"]["data_path"],
                                  "portfolios_by_year_market_cap.csv")
btmv_file_path = path.join(config["config_vars"]["data_path"],
                             "portfolios_by_year_book-to-market-value.csv")
returns_file_path = path.join(config["config_vars"]["data_path"],
                               "portfolios_by_year_returns.csv")
illiquidity_file_path = path.join(config["config_vars"]["data_path"],
                                   "portfolios_by_year_illiquidity.csv")
beta_file_path = path.join(config["config_vars"]["data_path"],
                            "portfolios_by_year_beta.csv")


def agg_df(file_path, periodicity, indicator):

    def agg_indicator(group):
        return group[indicator].tolist()

    def agg_terciles(group):
        t_1 = len(group[group["tercile"] == 1])
        t_2 = len(group[group["tercile"] == 2])
        t_3 = len(group[group["tercile"] == 3])
        return [t_1, t_2, t_3]

    df_agg = (
        pd.read_csv(file_path)
        .groupby("ticker", as_index=False)
        .apply(
            lambda x: pd.Series({
                f"Total tercis {indicator} (1, 2, 3)": agg_terciles(x),
                f"Historico {indicator}": agg_indicator(x),
            }),
            include_groups=False,
        )
    )
    return df_agg


def _load_latest_tercile(file_path: str, indicator: str) -> pd.DataFrame:
    """Read pre-built portfolio CSV and return the most-recent year's tercile assignments."""
    df = pd.read_csv(file_path)
    date_col = "data" if "data" in df.columns else df.columns[1]
    max_val = df[date_col].max()
    return (
        df[df[date_col] == max_val][["ticker", "tercile"]]
        .rename(columns={"tercile": f"Tercil {indicator} Atual"})
        .drop_duplicates("ticker")
    )


@st.cache_data()
def build_agg_df():
    df_size_agg = agg_df(market_cap_data_feed, "month", "market_cap")
    df_value_agg = agg_df(btmv_file_path, "month", "book-to-market-value")
    df_momentum_agg = agg_df(returns_file_path, "month", "returns")
    df_beta_agg = agg_df(beta_file_path, "month", "beta")
    df_illiq_agg = agg_df(illiquidity_file_path, "month", "illiquidity")

    # Read pre-built portfolio CSVs directly — avoids expensive rebuild
    portfolios_size = _load_latest_tercile(market_cap_data_feed, "market_cap")
    portfolios_value = _load_latest_tercile(btmv_file_path, "book-to-market-value")
    portfolios_momentum = _load_latest_tercile(returns_file_path, "returns")
    portfolios_illiquidity = _load_latest_tercile(illiquidity_file_path, "illiquidity")
    portfolios_beta = _load_latest_tercile(beta_file_path, "beta")

    lista_dfs = [
        df_size_agg, df_value_agg, df_momentum_agg,
        df_illiq_agg, df_beta_agg,
        portfolios_size, portfolios_value, portfolios_momentum,
        portfolios_illiquidity, portfolios_beta,
    ]
    all_tickers_agg = reduce(
        lambda left, right: left.merge(right, on="ticker", how="left"),
        lista_dfs,
    ).dropna()
    return all_tickers_agg


# ------------------------------------------------------------------ #
#  ECharts helpers                                                     #
# ------------------------------------------------------------------ #

def _accum_chart(dates, realized_pct, expected_pct) -> dict:
    def _fmt(v):
        return round(v, 2) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

    return {
        "backgroundColor": "#0E1117",
        "title": {
            "text": "Retorno Esperado Anualizado vs Retorno Realizado (12m à frente)",
            "textStyle": {"color": "#FFFFFF", "fontSize": 14},
        },
        "tooltip": {"trigger": "axis", "formatter": "{b}<br/>{a0}: {c0}%<br/>{a1}: {c1}%"},
        "legend": {
            "data": ["Retorno Realizado (12m à frente)", "Retorno Esperado Anualizado (APT)"],
            "textStyle": {"color": "#FFFFFF"},
        },
        "toolbox": {"feature": {"saveAsImage": {}}},
        "dataZoom": [
            {"show": True, "start": 0, "end": 100},
            {"type": "inside"},
        ],
        "xAxis": {
            "type": "category",
            "data": [d.strftime("%Y-%m") for d in dates],
            "axisLabel": {"color": "#AAAAAA", "rotate": 45},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#AAAAAA", "formatter": "{value}%"},
        },
        "series": [
            {
                "name": "Retorno Realizado (12m à frente)",
                "type": "line",
                "data": [_fmt(v) for v in realized_pct],
                "lineStyle": {"color": "#f5770c", "width": 2},
                "symbol": "none",
                "areaStyle": {"color": "rgba(245,119,12,0.08)"},
                "connectNulls": False,
            },
            {
                "name": "Retorno Esperado Anualizado (APT)",
                "type": "line",
                "data": [_fmt(v) for v in expected_pct],
                "lineStyle": {"color": "#4CAF50", "width": 2, "type": "dashed"},
                "symbol": "none",
            },
        ],
    }


def _attribution_chart(attribution: dict) -> dict:
    labels = [k for k in attribution if k != "Retorno Real"]
    values = [round(attribution[k] * 100, 3) for k in labels]
    bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in values]
    return {
        "backgroundColor": "#0E1117",
        "title": {"text": "Atribuição de Performance por Fator",
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
            "data": [
                {"value": v, "itemStyle": {"color": c}}
                for v, c in zip(values, bar_colors)
            ],
            "label": {"show": True, "formatter": "{c}%",
                      "color": "#FFFFFF", "position": "right"},
        }],
    }


# ------------------------------------------------------------------ #
#  Main simulation page                                                #
# ------------------------------------------------------------------ #

def simulate(dataframe: pd.DataFrame):

    elegibilidade_path = Path(__file__).resolve().parent.parent.parent / "data" / config["elegibilidade_final_file_path"]
    df_eleg = pd.read_csv(elegibilidade_path)
    df_eleg_latest = df_eleg[df_eleg["ano"] == df_eleg["ano"].max()]
    dataframe = dataframe[dataframe["ticker"].isin(df_eleg_latest["ticker"])]

    with st.container():
        st.html("<span class='any_container'></span>")
        st.markdown("# Simulador de Carteira por Fatores")
        st.markdown(
            "Selecione um ou mais ativos na tabela abaixo para montar sua carteira "
            "e analisar a exposição a fatores de risco."
        )

        # ---- stock selection table ----
        ativos_selection = st.dataframe(
            dataframe,
            column_config={
                "Historico market_cap": st.column_config.AreaChartColumn(
                    "Histórico Market Cap", width="medium", y_min=0, y_max=5_000_000),
                "Historico book-to-market-value": st.column_config.AreaChartColumn(
                    "Histórico BTMV", width="medium", y_min=-5, y_max=5),
                "Historico returns": st.column_config.AreaChartColumn(
                    "Histórico Momentum", width="medium", y_min=-2, y_max=2),
                "Historico illiquidity": st.column_config.AreaChartColumn(
                    "Histórico Iliquidez", width="medium", y_min=-1, y_max=1),
                "Historico beta": st.column_config.AreaChartColumn(
                    "Histórico Beta", width="medium", y_min=-1, y_max=5),
                "Total tercis market_cap (1, 2, 3)": st.column_config.BarChartColumn(
                    "Tercis Market Cap", width="medium", y_min=0, y_max=30),
                "Total tercis book-to-market-value (1, 2, 3)": st.column_config.BarChartColumn(
                    "Tercis BTMV", width="medium", y_min=0, y_max=30),
                "Total tercis returns (1, 2, 3)": st.column_config.BarChartColumn(
                    "Tercis Momentum", width="medium", y_min=0, y_max=30),
                "Total tercis illiquidity (1, 2, 3)": st.column_config.BarChartColumn(
                    "Tercis Iliquidez", width="medium", y_min=0, y_max=30),
                "Total tercis beta (1, 2, 3)": st.column_config.BarChartColumn(
                    "Tercis Beta", width="medium", y_min=0, y_max=30),
            },
            on_select="rerun",
            selection_mode=["multi-row"],
            hide_index=True,
        )

        indices = ativos_selection.selection["rows"]
        if not indices:
            st.info("Selecione pelo menos um ativo na tabela para iniciar a simulação.")
            return

        tickers = dataframe.iloc[indices]["ticker"].tolist()
        st.success(f"Carteira selecionada: **{', '.join(tickers)}**")

        # ---- configuration ----
        st.markdown("---")
        st.markdown("### Configurações da Simulação")
        window_months = st.radio(
            "Janela para cálculo dos betas (rolling OLS)",
            options=[12, 24, 36],
            format_func=lambda x: f"{x} meses",
            horizontal=True,
        )

        bt = Backtest()

        with st.spinner("Calculando betas e retorno esperado…"):
            try:
                betas = bt.get_factor_betas(tickers, window_months=window_months)
            except ValueError as e:
                st.error(str(e))
                return

        expected_return = bt.get_expected_return(betas)

        # ---- current expected return metric ----
        st.markdown("---")
        st.markdown("### Retorno Esperado Atual (APT)")
        cols_metrics = st.columns(len(FACTOR_DISPLAY) + 2)
        cols_metrics[0].metric(
            "Retorno Esperado Anual",
            f"{expected_return * 100:.2f}%",
            help="E(R) = Rf + Σ(βᵢ × prêmio médio do fator)",
        )
        cols_metrics[1].metric("R²", f"{betas['r2']:.3f}")
        for col, factor in zip(cols_metrics[2:], FACTOR_DISPLAY):
            col.metric(f"β {factor}", f"{betas.get(factor, 0):.3f}")

        # ---- historical returns chart ----
        st.markdown("---")
        st.markdown("### Histórico: Retorno Esperado Anualizado vs Retorno Realizado (12m à frente)")

        with st.spinner("Calculando rolling OLS e retornos históricos…"):
            try:
                hist_df = bt.get_historical_expected_vs_realized(tickers, window_months=window_months)
            except ValueError as e:
                st.error(str(e))
                return

        dates = hist_df["ano_mes"].dt.to_timestamp().tolist()
        expected_pct = (hist_df["expected_annual"] * 100).tolist()
        realized_pct = hist_df["realized_annual"].where(hist_df["realized_annual"].notna(), other=None)
        realized_pct = (realized_pct * 100).tolist()

        st_echarts(
            options=_accum_chart(dates, realized_pct, expected_pct),
            height=420,
        )
        st.caption(
            "Retorno Esperado: (1 + CDI_t + Σ β_t · prêmio_t)¹² − 1, "
            "com betas estimados por rolling OLS sobre os últimos "
            f"{window_months} meses. "
            "Retorno Realizado: retorno composto dos 12 meses seguintes (vazio nos últimos 12 pontos)."
        )

        # ---- performance attribution ----
        st.markdown("---")
        st.markdown("### Atribuição de Performance")

        min_date = hist_df["ano_mes"].dt.to_timestamp().min().date()
        max_date = hist_df["ano_mes"].dt.to_timestamp().max().date()
        default_start = max_date - timedelta(days=365)
        if default_start < min_date:
            default_start = min_date

        attr_cols = st.columns(2)
        attr_start = attr_cols[0].date_input(
            "Data de início", value=default_start,
            min_value=min_date, max_value=max_date,
        )
        attr_end = attr_cols[1].date_input(
            "Data de fim", value=max_date,
            min_value=min_date, max_value=max_date,
        )

        if attr_start >= attr_end:
            st.warning("A data de início deve ser anterior à data de fim.")
            return

        attribution = bt.get_performance_attribution(
            tickers, betas,
            start_date=str(attr_start),
            end_date=str(attr_end),
        )

        real_ret = attribution.pop("Retorno Real")
        st.metric(
            "Retorno Real da Carteira no Período",
            f"{real_ret * 100:.2f}%",
        )

        st_echarts(options=_attribution_chart(attribution), height=420)

        # attribution table
        attr_df = pd.DataFrame([
            {"Componente": k, "Contribuição (%)": round(v * 100, 3)}
            for k, v in attribution.items()
        ])
        with st.expander("Ver tabela de atribuição"):
            st.dataframe(attr_df, hide_index=True)


if __name__ == "__main__":
    agg = build_agg_df()
    simulate(agg)
