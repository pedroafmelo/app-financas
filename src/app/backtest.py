# coding: UTF-8 -*-
""" Import Modules """

import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from datetime import date, datetime, timedelta
from pathlib import Path
from bcb import sgs
from utils import sanitize_df, get_config, get_bcb

config = get_config()


FACTOR_RENAME = {
    "market_factor": "Mkt-Rf",
    "factor_size": "SMB",
    "factor_value": "HML",
    "factor_momentum": "WML",
    "factor_illiquidity": "IML",
    "factor_bab": "BAB",
}

FACTOR_TIME_SERIES = {
    "Mkt-Rf": {"file": "time_series_mkt_return", "factor_col": "market_factor", "tercile_cols": None},
    "SMB":    {"file": "time_series_size",        "factor_col": "factor_size",   "tercile_cols": ["retorno_carteira_tercil_1", "retorno_carteira_tercil_2", "retorno_carteira_tercil_3"]},
    "HML":    {"file": "time_series_value",       "factor_col": "factor_value",  "tercile_cols": ["retorno_carteira_tercil_1", "retorno_carteira_tercil_2", "retorno_carteira_tercil_3"]},
    "WML":    {"file": "time_series_momentum",    "factor_col": "factor_momentum", "tercile_cols": ["retorno_carteira_tercil_1", "retorno_carteira_tercil_2", "retorno_carteira_tercil_3"]},
    "IML":    {"file": "time_series_illiquidity", "factor_col": "factor_illiquidity", "tercile_cols": ["retorno_carteira_tercil_1", "retorno_carteira_tercil_2", "retorno_carteira_tercil_3"]},
    "BAB":    {"file": "time_series_bab",         "factor_col": "factor_bab",    "tercile_cols": ["retorno_carteira_tercil_1", "retorno_carteira_tercil_2", "retorno_carteira_tercil_3"]},
}


class Backtest:

    """ Backtesting analysis class """

    BG = "#fcfdff"
    PRIMARY = "#1c3f5f"
    TEXT = "#000000"
    GRID = "#333333"
    PALETTE = ["#f5770c", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    def __init__(self):
        self.project_path = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_path / "data"

        df_fact = pd.read_csv(self.data_path / "all_factors.csv")
        self.start_date = df_fact["data"].iloc[0]
        self.final_date = df_fact["data"].iloc[-1]

        self.stress_days = {
            "2008-09-15": "Lehman Brothers quebra (Crise 2008)",
            "2010-05-06": "Flash Crash nos EUA",
            "2011-08-08": "Rebaixamento dívida EUA (S&P)",
            "2013-06-17": "Protestos de Junho no Brasil",
            "2014-10-26": "Reeleição Dilma Rousseff",
            "2015-03-13": "Crise política Brasil (impeachment)",
            "2017-05-17": "Joesley Day (delação JBS)",
            "2018-05-18": "Greve dos caminhoneiros",
            "2018-10-28": "Eleição Jair Bolsonaro",
            "2020-03-09": "Circuit breaker COVID-19",
            "2022-10-30": "Eleição Lula",
            "2023-01-08": "Invasão dos Três Poderes",
        }

    def __str__(self):
        return "Backtesting Analysis Class"

    def __repr__(self):
        return "Backtesting Analysis Class"


    def get_data(self, file_name: str, sep=",", feed=False) -> pd.DataFrame:
        try:
            if feed:
                path = file_name
                df = pd.read_csv(path, encoding="latin")
            else:
                path = self.data_path / f"{file_name}.csv"
                df = pd.read_csv(path, encoding="latin", sep=sep)
            if df.empty:
                raise pd.errors.EmptyDataError()
        except Exception as error:
            raise Exception(f"Não foi possível ler o arquivo {file_name}: {error}") from error
        return df

    def get_rfr_series(self, period="monthly"):
        if period == "monthly":
            fmt = "%m/%Y"
            filename = "cdi_mensal"
        else:
            fmt = "%d/%m/%Y"
            filename = "cdi_diario"

        df_cdi = (
            get_bcb(filename, start="2008-01-01", end=datetime.today())
            .set_axis(["data", "cdi"], axis="columns")
            .assign(
                cdi=lambda x: x["cdi"] / 100,
                data=lambda x: pd.to_datetime(x["data"], format=fmt),
            )
        )
        return df_cdi[["data", "cdi"]]

    def get_ibov_series(self):
        ibov = (
            sanitize_df(
                self.get_data(config["ibovespa_data_feed"], feed=True).set_axis(["ticker", "data", "close"], axis="columns"),
                cols_to_cast_float=["close"],
                cols_to_cast_int=[],
                col_to_cast_date="",
            )
            .assign(data=lambda x: pd.to_datetime(x["data"], format="%d/%m/%Y", errors="coerce"))
            .dropna(subset=["data", "close"])
            .set_index("data")
        )
        ibov_ret = ibov["close"].dropna().pct_change().dropna()
        return ibov_ret.reset_index().rename(columns={"close": "ibov"})

    def get_monthly_factor_returns(self) -> pd.DataFrame:
        """Aggregates daily all_factors.csv to monthly compound returns."""
        factors_df = (
            pd.read_csv(self.data_path / "all_factors.csv")
            .rename(columns=FACTOR_RENAME)
            .assign(data=lambda x: pd.to_datetime(x["data"]))
        )
        factor_cols = list(FACTOR_RENAME.values())
        monthly = (
            factors_df
            .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
            .groupby("ano_mes")[factor_cols]
            .agg(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )
        return monthly

    def get_portfolio_monthly_returns(self, tickers: list, cotacoes: pd.DataFrame = None) -> pd.DataFrame:
        """Equal-weighted portfolio monthly returns from closing prices."""
        
        if cotacoes is None:
            cotacoes = sanitize_df(
                self.get_data(config["cotacoes_liquidez_data_feed"], feed=True)
                .set_axis(["ticker", "data", "close", "qt_negs", "vol_negociado"], axis="columns"),
                cols_to_cast_float=["close"],
                cols_to_cast_int=[],
                col_to_cast_date="data",
            )[["ticker", "data", "close"]].dropna(subset="close")

        cotacoes = cotacoes[cotacoes["ticker"].isin(tickers)]
        if cotacoes.empty:
            raise ValueError("Nenhum dado de preço encontrado para os ativos selecionados.")

        monthly = (
            cotacoes
            .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
            .sort_values("data")
            .groupby(["ticker", "ano_mes"])
            .last()
            .reset_index()
            .assign(ret=lambda x: x.groupby("ticker")["close"].pct_change())
            .dropna(subset="ret")
        )

        portfolio = (
            monthly.groupby("ano_mes")["ret"]
            .mean()
            .reset_index()
            .rename(columns={"ret": "retorno_carteira"})
        )
        return portfolio

    def get_factor_betas(self, tickers: list, window_months: int = 12, 
                         monthly_factors=None, portfolio=None, rfr=None) -> dict:
        """OLS betas of equal-weighted portfolio vs factor returns."""
        factor_cols = list(FACTOR_RENAME.values())

        if portfolio is None:
            portfolio = self.get_portfolio_monthly_returns(tickers)
        
        if monthly_factors is None:    
            monthly_factors = self.get_monthly_factor_returns()
        
        if rfr is None:
            rfr = (
                self.get_rfr_series("monthly")
                .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
            )

        merged = (
            portfolio
            .merge(monthly_factors, on="ano_mes", how="inner")
            .merge(rfr[["ano_mes", "cdi"]], on="ano_mes", how="left")
            .fillna({"cdi": 0})
            .tail(window_months)
        )

        if len(merged) < max(6, window_months // 3):
            raise ValueError(f"Dados insuficientes para calcular betas ({len(merged)} meses).")

        Y = merged["retorno_carteira"] - merged["cdi"]
        X = sm.add_constant(merged[[fac for fac in factor_cols if fac in monthly_factors.columns]])
        model = sm.OLS(Y, X, missing="drop").fit()

        betas = {col: float(model.params.get(col, 0.0)) for col in factor_cols if col in monthly_factors.columns}
        betas["alpha"] = float(model.params.get("const", 0.0))
        betas["r2"] = float(model.rsquared)
        betas["p_values"] = {col: float(model.pvalues.get(col, 1.0)) for col in factor_cols if col in monthly_factors.columns}
        return betas

    def get_expected_return(self, betas: dict, window_months: int=12,
                            rfr_monthly = None, monthly_factors = None) -> float:
        """E(R) annualized = (1 + Rf_monthly + Σ(βi × mean_factor_monthly))^12 − 1."""
        if monthly_factors is None:
            monthly_factors = self.get_monthly_factor_returns()
        monthly_factors = monthly_factors.tail(window_months)
        factor_cols = [fac for fac in list(FACTOR_RENAME.values()) if fac in monthly_factors.columns]
        mean_premia = monthly_factors[factor_cols].mean()
        if rfr_monthly is None:
            rfr_monthly = self.get_rfr_series("monthly")
        rfr_monthly = rfr_monthly.tail(window_months)["cdi"].mean()
        factor_contribution = sum(betas.get(f, 0.0) * mean_premia[f] for f in factor_cols)
        return float(((1 + rfr_monthly + factor_contribution) ** 12) - 1)

    def get_historical_expected_vs_realized(self, tickers: list, window_months: int = 24) -> pd.DataFrame:
        """Rolling OLS betas → annualized expected return at each t vs forward 12-month realized return.

        Returns a DataFrame with columns:
          ano_mes          — Period index
          expected_annual  — APT expected return annualized: (1 + Rf_t + Σβ_t·f_t)^12 − 1
          realized_annual  — Compound return of the NEXT 12 months (NaN for last 12 rows)
        """
        factor_cols = list(FACTOR_RENAME.values())

        portfolio = self.get_portfolio_monthly_returns(tickers)
        monthly_factors = self.get_monthly_factor_returns()
        rfr = (
            self.get_rfr_series("monthly")
            .assign(ano_mes=lambda x: x["data"].dt.to_period("M"))
        )

        merged = (
            portfolio
            .merge(monthly_factors, on="ano_mes", how="inner")
            .merge(rfr[["ano_mes", "cdi"]], on="ano_mes", how="left")
            .fillna({"cdi": 0})
            .sort_values("ano_mes")
            .reset_index(drop=True)
        )

        if len(merged) < window_months:
            raise ValueError(
                f"Dados insuficientes para rolling OLS: {len(merged)} meses < janela {window_months}."
            )

        Y = merged["retorno_carteira"] - merged["cdi"]
        X = sm.add_constant(merged[factor_cols])

        rolling_params = RollingOLS(Y, X, window=window_months).fit().params
        # rolling_params: DataFrame (n_obs × k), NaN for first window_months-1 rows

        beta_mat = rolling_params[factor_cols].values        # shape (n, 6)
        factor_ret_mat = merged[factor_cols].values          # shape (n, 6)
        dot = np.einsum("ij,ij->i", beta_mat, factor_ret_mat)
        expected_monthly = merged["cdi"].values + dot
        expected_annual = (1 + expected_monthly) ** 12 - 1

        port_ret = merged["retorno_carteira"].values
        n = len(port_ret)
        realized_annual = np.full(n, np.nan)
        for i in range(n - 12):
            realized_annual[i] = float((1 + port_ret[i + 1: i + 13]).prod() - 1)

        result = merged[["ano_mes"]].copy()
        result["expected_annual"] = expected_annual
        result["realized_annual"] = realized_annual

        valid = rolling_params[factor_cols].notna().all(axis=1)
        return result[valid].reset_index(drop=True)

    def get_performance_attribution(self, tickers: list, betas: dict,
                                     start_date: str, end_date: str) -> dict:
        """Returns factor-by-factor attribution and alpha for a given period."""
        factor_cols = list(FACTOR_RENAME.values())

        monthly_factors = self.get_monthly_factor_returns().assign(
            data=lambda x: x["ano_mes"].dt.to_timestamp()
        )
        sd, ed = pd.to_datetime(start_date), pd.to_datetime(end_date)
        period_factors = monthly_factors[
            (monthly_factors["data"] >= sd) & (monthly_factors["data"] <= ed)
        ]
        factor_cumret = (1 + period_factors[factor_cols]).prod() - 1

        portfolio = self.get_portfolio_monthly_returns(tickers).assign(
            data=lambda x: x["ano_mes"].dt.to_timestamp()
        )
        period_port = portfolio[(portfolio["data"] >= sd) & (portfolio["data"] <= ed)]
        portfolio_cumret = float((1 + period_port["retorno_carteira"]).prod() - 1)

        rfr = self.get_rfr_series("monthly").assign(
            data=lambda x: pd.to_datetime(x["data"])
        )
        period_rfr = rfr[(rfr["data"] >= sd) & (rfr["data"] <= ed)]
        rfr_cumret = float((1 + period_rfr["cdi"]).prod() - 1)

        attribution = {"CDI (Rf)": rfr_cumret}
        total_factor = 0.0
        for f in factor_cols:
            contrib = betas.get(f, 0.0) * float(factor_cumret.get(f, 0.0))
            attribution[f] = contrib
            total_factor += contrib

        attribution["Alfa"] = portfolio_cumret - rfr_cumret - total_factor
        attribution["Retorno Real"] = portfolio_cumret
        return attribution

    # ------------------------------------------------------------------ #
    #  Basic analytics (kept from original)                               #
    # ------------------------------------------------------------------ #

    def get_accum_return(self, data: pd.DataFrame, col_name: str):
        try:
            return (1 + data[col_name]).cumprod() - 1
        except Exception as error:
            raise Exception(f"Não foi possível construir o retorno acumulado: {error}") from error

    def drawdown(self, data: pd.DataFrame, col_name: str):
        accum = self.get_accum_return(data, col_name)
        df_dd = data[["data"]].copy()
        df_dd["accum"] = accum
        df_dd["peak"] = accum.cummax()
        df_dd["drawdown"] = (df_dd["accum"] + 1) / (df_dd["peak"] + 1) - 1
        return df_dd

    def get_var(self, data: pd.DataFrame, col_name: str, confidence: float = 0.95) -> float:
        returns = data[col_name].dropna()
        return float(np.percentile(returns, (1 - confidence) * 100))

    def stress_test(self, data: pd.DataFrame, col_name: str) -> dict:
        """Cumulative return D → D+2 for each stress event."""
        df = data.assign(data=lambda x: pd.to_datetime(x["data"])).reset_index(drop=True)
        results = {}
        for date_str, event_name in self.stress_days.items():
            event_date = pd.to_datetime(date_str)
            nearby = df[df["data"] >= event_date].head(3)
            if nearby.empty:
                continue
            cumret = float((1 + nearby[col_name]).prod() - 1)
            results[event_name] = cumret
        return results

    def get_rolling_sharpe(self, df: pd.DataFrame, col_name: str, window: int = 252):
        ret = df[col_name]
        sharpe = ret.rolling(window).apply(
            lambda x: (((1 + x.mean()) ** 252) - 1) / (x.std() * np.sqrt(252))
            if x.std() > 0 else np.nan,
            raw=True,
        )
        result = df[["data"]].copy()
        result["sharpe"] = sharpe
        return result[["data", "sharpe"]]

    def get_basic_stats(self, df: pd.DataFrame, factor: str):
        try:
            retornos = df[factor]
            rfr = ((1 + self.get_rfr_series()["cdi"].mean()) ** 12) - 1
            ibov = self.get_ibov_series()

            factor_ibov = df[["data", factor]].merge(ibov, on="data", how="left")
            monthly_factor_ibov = (
                factor_ibov
                .assign(
                    data=pd.to_datetime(factor_ibov["data"]),
                    ano_mes=lambda x: x["data"].dt.to_period("M"),
                )
                .sort_values("data")
                .groupby("ano_mes")[["ibov", factor]]
                .apply(lambda x: (1 + x).prod() - 1, include_groups=False)
                .reset_index()
            )

            media_diaria = retornos.mean()
            media_anualizada = ((1 + media_diaria) ** 252) - 1
            vol_diaria = retornos.std()
            vol_anualizada = vol_diaria * np.sqrt(252)
            sharpe_medio = (media_anualizada - rfr) / vol_anualizada
            ret_acum_total = self.get_accum_return(df, factor).iloc[-1]
            total_anos = pd.to_datetime(df["data"]).dt.year.nunique()
            retorno_medio_anual = ((1 + ret_acum_total) ** (1 / total_anos)) - 1
            ret_positivos = retornos[retornos > 0]
            ret_negativos = retornos[retornos < 0]
            pct_dias_positivos = (retornos > 0).mean()
            pct_dias_negativos = (retornos < 0).mean()
            media_positivos = ((1 + ret_positivos.mean()) ** 252) - 1
            media_negativos = ((1 + ret_negativos.mean()) ** 252) - 1
            days_win_ibov = len(factor_ibov[factor_ibov[factor] > factor_ibov["ibov"]]) / len(factor_ibov)
            months_win_ibov = len(monthly_factor_ibov[monthly_factor_ibov[factor] > monthly_factor_ibov["ibov"]]) / len(monthly_factor_ibov)
            max_dd = self.drawdown(df, factor)["drawdown"].min()

            return {
                "retorno_medio_diario": media_diaria,
                "retorno_medio_anual": media_anualizada,
                "vol_media_diaria": vol_diaria,
                "vol_media_anualizada": vol_anualizada,
                "sharpe": sharpe_medio,
                "retorno_acumulado": ret_acum_total,
                "retorno_medio_ao_ano": retorno_medio_anual,
                "max_dd": max_dd,
                "pct_positivos": pct_dias_positivos,
                "pct_negativos": pct_dias_negativos,
                "media_retorno_positivos": media_positivos,
                "media_retorno_negativos": media_negativos,
                "pct_dias_ganhos_ibov": days_win_ibov,
                "pct_mes_ganhos_ibov": months_win_ibov,
            }
        except Exception as error:
            raise Exception(f"Não foi possível calcular as estatísticas básicas: {error}") from error

    # ------------------------------------------------------------------ #
    #  Matplotlib dark theme helpers                                       #
    # ------------------------------------------------------------------ #

    def _setup_style(self):
        plt.rcParams.update({
            "figure.facecolor": self.BG,
            "axes.facecolor": self.BG,
            "axes.edgecolor": self.GRID,
            "axes.labelcolor": self.TEXT,
            "text.color": self.TEXT,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "grid.color": self.GRID,
            "grid.alpha": 0.5,
            "legend.facecolor": "#1a1e2e",
            "legend.edgecolor": self.GRID,
            "legend.labelcolor": self.TEXT,
        })

    def _fig_to_bytes(self, fig) -> BytesIO:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=self.BG)
        buf.seek(0)
        plt.close(fig)
        return buf

    # ------------------------------------------------------------------ #
    #  Chart methods (return BytesIO for PDF embedding)                   #
    # ------------------------------------------------------------------ #

    def plot_accum_return(self, data: pd.DataFrame, col_name: str,
                           benchmarks: dict = None) -> BytesIO:
        self._setup_style()
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = pd.to_datetime(data["data"])
        accum = self.get_accum_return(data, col_name) * 100

        ax.plot(dates, accum, color=self.PRIMARY, linewidth=2, label=col_name)
        ax.fill_between(dates, accum, alpha=0.08, color=self.PRIMARY)

        if benchmarks:
            for (name, df), color in zip(benchmarks.items(), ["#4CAF50", "#2196F3"]):
                b_accum = self.get_accum_return(df, name) * 100
                ax.plot(pd.to_datetime(df["data"]), b_accum,
                        color=color, linewidth=1.5, label=name, linestyle="--")

        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_title(f"Retorno Acumulado — {col_name}", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(framealpha=0.3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_accum_return_multiple(self, data: pd.DataFrame, factor_cols: list) -> BytesIO:
        self._setup_style()
        fig, ax = plt.subplots(figsize=(14, 6))
        dates = pd.to_datetime(data["data"])
        for col, color in zip(factor_cols, self.PALETTE):
            accum = (1 + data[col]).cumprod() - 1
            ax.plot(dates, accum * 100, color=color, linewidth=1.8, label=col)
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_title("Retorno Acumulado — Todos os Fatores", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(framealpha=0.3, ncol=3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_drawdown(self, data: pd.DataFrame, col_name: str) -> BytesIO:
        self._setup_style()
        df_dd = self.drawdown(data, col_name)
        dates = pd.to_datetime(df_dd["data"])
        dd = df_dd["drawdown"] * 100

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(dates, dd, 0, color="#8b0000", alpha=0.7)
        ax.plot(dates, dd, color="#ff4444", linewidth=1)

        min_idx = dd.idxmin()
        ax.annotate(f"Max DD: {dd[min_idx]:.2f}%",
                    xy=(dates[min_idx], dd[min_idx]),
                    xytext=(20, 15), textcoords="offset points",
                    color=self.TEXT, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=self.TEXT))

        ax.set_title(f"Drawdown — {col_name}", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_rolling_returns(self, data: pd.DataFrame, col_name: str,
                              window_months: int = 12) -> BytesIO:
        self._setup_style()
        monthly = (
            data.assign(ano_mes=lambda x: pd.to_datetime(x["data"]).dt.to_period("M"))
            .groupby("ano_mes")[col_name]
            .agg(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )
        monthly["rolling"] = (
            monthly[col_name]
            .rolling(window_months)
            .apply(lambda x: (1 + x).prod() - 1, raw=True) * 100
        )
        monthly = monthly.dropna(subset="rolling")
        dates = monthly["ano_mes"].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(14, 4))
        bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in monthly["rolling"]]
        ax.bar(dates, monthly["rolling"], color=bar_colors, width=25, alpha=0.85)
        ax.axhline(0, color="#AAAAAA", linewidth=0.8)
        ax.set_title(f"Retorno Rolling {window_months} meses — {col_name}",
                     fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_rolling_sharpe(self, data: pd.DataFrame, col_name: str,
                             window: int = 252) -> BytesIO:
        self._setup_style()
        ret = data[col_name]
        sharpe = ret.rolling(window).apply(
            lambda x: (((1 + x.mean()) ** 252) - 1) / (x.std() * np.sqrt(252))
            if x.std() > 0 else np.nan,
            raw=True,
        )
        dates = pd.to_datetime(data["data"])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(dates, sharpe, color=self.PRIMARY, linewidth=1.5)
        ax.fill_between(dates, sharpe, 0,
                        where=sharpe >= 0, alpha=0.15, color="#4CAF50")
        ax.fill_between(dates, sharpe, 0,
                        where=sharpe < 0, alpha=0.15, color="#f44336")
        ax.axhline(0, color="#AAAAAA", linewidth=0.8)
        ax.axhline(1, color="#4CAF50", linewidth=0.8, linestyle=":", alpha=0.6,
                   label="Sharpe = 1")
        ax.set_title(f"Sharpe Ratio Rolling ({window} dias) — {col_name}",
                     fontsize=14, fontweight="bold")
        ax.legend(framealpha=0.3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_rolling_sharpe_multiple(self, data: pd.DataFrame,
                                      factor_cols: list, window: int = 252) -> BytesIO:
        self._setup_style()
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = pd.to_datetime(data["data"])
        for col, color in zip(factor_cols, self.PALETTE):
            sharpe = data[col].rolling(window).apply(
                lambda x: (((1 + x.mean()) ** 252) - 1) / (x.std() * np.sqrt(252))
                if x.std() > 0 else np.nan,
                raw=True,
            )
            ax.plot(dates, sharpe, color=color, linewidth=1.5, label=col, alpha=0.85)
        ax.axhline(0, color="#AAAAAA", linewidth=0.8)
        ax.axhline(1, color="#555", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(f"Sharpe Ratio Rolling ({window} dias) — Todos os Fatores",
                     fontsize=14, fontweight="bold")
        ax.legend(framealpha=0.3, ncol=3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_monthly_heatmap(self, data: pd.DataFrame, col_name: str) -> BytesIO:
        self._setup_style()
        monthly = (
            data.assign(
                year=lambda x: pd.to_datetime(x["data"]).dt.year,
                month=lambda x: pd.to_datetime(x["data"]).dt.month,
            )
            .groupby(["year", "month"])[col_name]
            .agg(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )
        pivot = monthly.pivot(index="year", columns="month", values=col_name) * 100
        month_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                        "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        pivot.columns = [month_labels[c - 1] for c in pivot.columns]

        fig_h = max(5, len(pivot) * 0.6)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        sns.heatmap(pivot, annot=True, fmt=".1f", center=0,
                    cmap=sns.diverging_palette(10, 130, as_cmap=True),
                    ax=ax, linewidths=0.4, linecolor="#333",
                    annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.6, "label": "%"})
        ax.set_title(f"Retorno Mensal (%) — {col_name}", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_histogram(self, data: pd.DataFrame, col_name: str,
                        var_95: float = None) -> BytesIO:
        self._setup_style()
        returns = data[col_name].dropna() * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        n, bins, patches = ax.hist(returns, bins=80, color=self.PRIMARY,
                                   alpha=0.75, edgecolor="none")
        for patch, left_edge in zip(patches, bins):
            if left_edge < 0:
                patch.set_facecolor("#f44336")

        ax.axvline(returns.mean(), color="#4CAF50", linestyle="--", linewidth=1.5,
                   label=f"Média: {returns.mean():.3f}%")
        if var_95 is not None:
            ax.axvline(var_95 * 100, color="#ff1744", linestyle="--", linewidth=2,
                       label=f"VaR 95%: {var_95 * 100:.3f}%")

        ax.set_title(f"Histograma de Retornos — {col_name}", fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax.legend(framealpha=0.3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_stress_test(self, stress_results: dict) -> BytesIO:
        self._setup_style()
        events = list(stress_results.keys())
        returns_pct = [v * 100 for v in stress_results.values()]
        bar_colors = ["#4CAF50" if r >= 0 else "#f44336" for r in returns_pct]
        short = [e[:38] + "…" if len(e) > 38 else e for e in events]

        fig, ax = plt.subplots(figsize=(14, max(5, len(events) * 0.65)))
        bars = ax.barh(short, returns_pct, color=bar_colors, alpha=0.85, height=0.6)
        ax.axvline(0, color="#AAAAAA", linewidth=1)
        for bar, val in zip(bars, returns_pct):
            offset = 0.08 if val >= 0 else -0.08
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", ha=ha, fontsize=9)

        ax.set_title("Stress Test - Retorno Acumulado D -> D+2", fontsize=14, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_tercile_accum(self, data: pd.DataFrame, tercile_cols: list,
                            factor_col: str) -> BytesIO:
        self._setup_style()
        dates = pd.to_datetime(data["data"])
        t_colors = ["#2196F3", "#FF9800", "#4CAF50"]
        t_labels = ["Tercil 1", "Tercil 2", "Tercil 3"]

        fig, ax = plt.subplots(figsize=(14, 5))
        for col, color, label in zip(tercile_cols, t_colors, t_labels):
            accum = (1 + data[col]).cumprod() - 1
            ax.plot(dates, accum * 100, color=color, linewidth=1.5, label=label)

        factor_accum = (1 + data[factor_col]).cumprod() - 1
        ax.plot(dates, factor_accum * 100, color=self.PRIMARY, linewidth=2.5,
                linestyle="--", label=f"Fator ({factor_col})")

        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_title("Retorno Acumulado por Tercil", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(framealpha=0.3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_tercile_bars(self, data: pd.DataFrame, tercile_cols: list) -> BytesIO:
        self._setup_style()
        labels = ["Tercil 1", "Tercil 2", "Tercil 3"]
        values_pct = [float((1 + data[c]).prod() - 1) * 100 for c in tercile_cols]
        t_colors = ["#2196F3", "#FF9800", "#4CAF50"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values_pct, color=t_colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values_pct):
            y_pos = val + 1 if val >= 0 else val - 4
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.1f}%", ha="center", fontsize=12)

        ax.set_title("Retorno Total Acumulado por Tercil", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def plot_correlation_heatmap(self, data: pd.DataFrame, factor_cols: list) -> BytesIO:
        self._setup_style()
        corr = data[factor_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", center=0,
                    cmap=sns.diverging_palette(10, 130, as_cmap=True),
                    ax=ax, mask=mask, linewidths=0.5,
                    vmin=-1, vmax=1, annot_kws={"size": 11},
                    cbar_kws={"shrink": 0.6})
        ax.set_title("Matriz de Correlação — Fatores de Risco",
                     fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        return self._fig_to_bytes(fig)