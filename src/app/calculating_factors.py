# -*- coding: UTF-8 -*-
""" Import Modules """

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from os import path
from typing import Literal
from functools import reduce
import warnings

# local imports
from utils import get_config, sanitize_df, update_config, get_bcb
warnings.filterwarnings("ignore")

config = get_config()


class RiskFactors:

    def __init__(self) -> None:
        """ Instanciates Risk Factors Class """

        self.date = date.today()
        
        self.indicators = [
            "market_cap", "book-to-market-value", 
            "returns", "illiquidity", "beta"
        ]

        self.factors = [
            "market", "size", "value", "momentum", 
            "illiquidity", "bab"
        ]

        self.months_dict = {
                    "Jan": "01",
                    "Fev": "02",
                    "Mar": "03",
                    "Abr": "04",
                    "Mai": "05",
                    "Jun": "06",
                    "Jul": "07",
                    "Ago": "08",
                    "Set": "09",
                    "Out": "10",
                    "Nov": "11",
                    "Dez": "12"
                }

    
    def get_elegibility(self) -> pd.DataFrame:
        """
        Gets stocks data and, for each year, get elegible 
        stocks based on nefin criteria
        
        :param self: instance param
        :return: dataframe with elegible stocks in each year
        :rtype: DataFrame
        """

        def get_trading_days_pct(group):
            """ Criteria 1: 80% of trading days in year t-1 """

            
            first_trade = group[group["vol_negociado"] > 0]["data"].min()
            total_days = group[group["data"] >= first_trade] # add weights to companys listed in that year

            min_500_volume = total_days[total_days["vol_negociado"] > 500]

            # agora faz mais sentido:
            # Antes eu fazia meio que: se a acao foi negociada mais de 80% dos dias, ok. 
            # mas na verdade é: a acao tem que ter 80% dos dias negociados com no min 500k negociado

            if len(total_days) == 0:
                return 0

            return int(len(min_500_volume)/len(total_days) > .8)

        def get_first_trading_date(group):
            """ Criteria 3: First trading day prior to December of t-1 """

            traded = group[group["qt_negocios"] > 0]
            if traded.empty:
                return np.nan
            
            return traded["data"].min().month

        try:

            df_indicadores = (sanitize_df(
                    pd.read_csv(
                        get_config()["indicadores_data_feed"], encoding="latin"
                        )
                        .set_axis(["ticker", "data", "market_cap", "preco_lucro", "lpa", "lucro", "qtd_acoes", 
                                   "qtd_acoes_ano", "pat_liq", "caixa", "data_balanco", "div_cp", 
                                   "div_lp", "close", "qt_negocios", "volume_negociado", "ac_discl_dt"], 
                                 axis="columns").filter(["ticker", "data", "pat_liq", "market_cap"], axis=1),
                    cols_to_cast_float=["pat_liq", "market_cap"],
                    cols_to_cast_int=[],
                    col_to_cast_date=None
                )
                .assign(
                    ano = lambda x: x["data"].str.slice(2).astype(int),
                    tri = lambda x: x["data"].str.slice(0, 2)
                )
                .query("tri == '2T'") # book value and market cap of 2T (t-1) should be > 0, avoiding book-to-market and size indicators outliers
                # i wish i could use the 4T, but when we analyze the PL of the company and this should be 
                # available only by march or april of year t+1, would impact our analysis.
                # does it make sense or not? the PL is just the photography of the company, right?
            )


            df = sanitize_df(
                    pd.read_csv(
                        get_config()["cotacoes_liquidez_data_feed"]
                        )
                        .set_axis(["ticker", "data", "close", "qt_negocios", "vol_negociado",], 
                                 axis="columns"),
                    cols_to_cast_float=["close", "vol_negociado"],
                    cols_to_cast_int=["qt_negocios"],
                    col_to_cast_date="data"
                )
        
            df_eleg = (
                df
                .assign(
                    ano = lambda x: x["data"].dt.year,
                    mes = lambda x: x["data"].dt.month
                )
                .groupby(["ano", "empresa", "ticker"], as_index=False)
                .apply(lambda x: pd.Series(
                        {
                            "total_volume_negociado": x["vol_negociado"].sum(),
                            "flag_trading_days": get_trading_days_pct(x),
                            "menor_data": get_first_trading_date(x)
                        }
                )
                )
                .sort_values(by="total_volume_negociado", ascending=False)
                .drop_duplicates(["ano", "empresa"]) 
                # this guarantees that the company selected stock is the one with higher traded volume in year t-1
                .merge(df_indicadores[["ticker", "ano", "pat_liq", "market_cap"]], 
                       on=["ticker", "ano"], how="left")
                .loc[
                    lambda x: (x["flag_trading_days"] == 1) &
                    (x["menor_data"] != 12) & 
                    (~x["menor_data"].isna()) & 
                    (x["pat_liq"] > 0) &
                    (x["market_cap"] > 0)
                ]
                .reset_index().drop("index", axis=1)
                .sort_values(["ano", "ticker"])
                .assign(
                    # shifting year
                    ano = lambda x: x["ano"] + 1
                )
            )

            update_config("config_vars", "elegibilidade_final_file_path", "elegibilidade_final.csv")

            df_eleg.to_csv(get_config()["elegibilidade_final_file_path"], index=False)

            return df_eleg
        
        except Exception as error:
            raise Exception(f"Couldn't get elegible stocks dataframe: {error}") from error

    
    def get_monthly_indicators(self) -> pd.DataFrame:
        """
        Build monthly indicators for
        momentum, illiquidity and bab factors
        
        :param self: instance param
        :return: monthly indicators dataframe
        :rtype: DataFrame
        """


        def get_momentum_df(close_df: pd.DataFrame) -> pd.DataFrame:

            try:

                df_momentum = (
                    close_df
                    .groupby("ticker")["close"]
                    .resample("ME")
                    .last()
                    .rename("close")
                    .reset_index()
                    .groupby("ticker")
                    .apply(
                        lambda x: pd.DataFrame({
                            "data": x["data"],
                            "momentum_12_2": x["close"].shift(2) / x["close"].shift(12) - 1
                        })
                    )
                    .reset_index(level=0)
                    .assign(
                        ano_mes = lambda x: x["data"].dt.strftime("%Y%m").astype(int)
                    )
                )


            except Exception as error:
                raise Exception(
                    f"Impossible to build momentum indicators: {error}"
                    ) from error
            
            return df_momentum
        

        def get_bab_df(close_df: pd.DataFrame) -> pd.DataFrame:

            try:

                ibov_data = (
                    sanitize_df(
                        (
                            pd.read_csv(get_config()["ibovespa_data_feed"])
                            .set_axis(["ticker", "data", "close"], axis="columns")
                        ),
                        cols_to_cast_float=["close"],
                        cols_to_cast_int=[],
                        col_to_cast_date="data"
                    )
                    .sort_values("data")
                    .set_index("data")
                    .dropna(subset="close")
                    .resample("ME")
                    .last()
                    .assign(
                        mkt_return = lambda x: x["close"].pct_change()
                    )
                    .dropna(subset="mkt_return")
                )[["mkt_return"]]

                # already in monthly returns
                cdi_data = (
                    get_bcb("cdi_mensal", start="2008-01-01", end=datetime.today())
                    .set_axis(["data", "rf_return"], axis="columns")
                    .assign(
                        data = lambda x: pd.to_datetime(x["data"], format="%m/%Y") + pd.offsets.MonthEnd(0),
                        rf_return = lambda x: x["rf_return"].astype(str).str.replace(",", ".").astype(float) / 100
                    )
                    .set_index("data")
                )

                assets_df = (
                        close_df.groupby("ticker", as_index=False)
                        .resample("ME")
                        .last()
                         .assign(
                             stock_return = lambda x: x.groupby("ticker")["close"].pct_change()
                         )
                         .dropna(subset="stock_return")
                )

                ibov_assets_closings = (
                    pd.merge(assets_df,ibov_data,right_index=True,left_index=True,how="inner")
                    .merge(cdi_data,right_index=True,left_index=True,how="inner")
                )

                excess = (
                    ibov_assets_closings
                    .assign(
                        stock_excess=lambda x: x["stock_return"] - x["rf_return"],
                        mkt_excess=lambda x: x["mkt_return"] - x["rf_return"],
                    )
                )

                rolling_cov = (
                    excess
                    .groupby("ticker")
                    .apply(lambda x: (
                        x["stock_excess"]
                        .rolling(24)
                        .cov(x["mkt_excess"])
                    ))
                )

                rolling_var = (
                    excess
                    .groupby("ticker")["mkt_excess"]
                    .rolling(24)
                    .var()
                )
                
                # the shift avoids look ahead
                df_beta = (
                    (rolling_cov / rolling_var)
                    .groupby(level=0)
                    .shift(1)
                    .reset_index()
                    .drop("level_1", axis=1).rename(columns={0: "beta_m24"})
                    .dropna(subset="beta_m24")
                )

                df_beta.columns = ["ticker", "data", "beta_m24"]

                df_bab = (
                    ibov_assets_closings.copy()
                    .reset_index()
                    .assign(
                        ano_mes = lambda x: x["data"].dt.strftime("%Y%m").astype(int)
                    )
                    [["data", "ticker", "ano_mes"]]
                    .merge(
                        df_beta,
                        on=["ticker", "data"],
                        how="left"
                    )
                    .sort_values(["data", "ticker"])
                    .set_index("data")
                    .groupby("ticker", as_index=False)
                    .resample("ME")
                    .last()
                )


            except Exception as error:
                raise Exception(
                    f"Impossible to build bab indicators: {error}"
                    ) from error
            
            return df_bab
        
        
        def get_liquidity_df(close_df: pd.DataFrame) -> pd.DataFrame:

            def acharya_pedersen_illi(group):

                dias_neg = len(group[group["vol_neg_mm"] > 0])

                if dias_neg == 0 :
                    return np.nan
                
                retornos = np.abs(group["returns"].to_numpy())

                vol_neg = group["vol_neg_mm"].to_numpy() / group["A_t"].to_numpy()

                illiquidity = min(0.25, (1/dias_neg * sum(retornos / vol_neg)))

                return illiquidity

            try:

                df_mkt_cap = (
                    sanitize_df(
                        (
                            pd.read_csv(config["market_cap_data_feed"])
                            .set_axis(["ticker", "data", "market_cap"], axis="columns")
                            .assign(
                                mes = lambda x: x["data"].str.slice(0, 3).map(self.months_dict),
                                ano = lambda x: x["data"].str.slice(4),
                                ano_mes = lambda x: x["ano"] + x["mes"],
                            )
                        ),
                         cols_to_cast_float=["market_cap"],
                         cols_to_cast_int=[],
                         col_to_cast_date=[]
                    )
                )

                mkt_cap_total = (
                    df_mkt_cap.groupby("ano_mes", as_index=True)["market_cap"].sum()
                    .reset_index()
                    .rename(columns={"market_cap": "mkt_cap_mercado"})
                    .sort_values("ano_mes")
                )

                jan_2000_val = mkt_cap_total.loc[
                    mkt_cap_total["ano_mes"] == "200001", "mkt_cap_mercado"
                ].values[0]

                mkt_cap_total = (
                    mkt_cap_total
                    .assign(
                        mkt_cap_mercado = lambda x: x["mkt_cap_mercado"].shift(),
                        mkt_cap_jan_2000 = lambda x: jan_2000_val,
                        A_t = lambda x: x["mkt_cap_mercado"] / x["mkt_cap_jan_2000"]
                    )
                    .dropna(subset=["A_t"])
                )

                df_daily = (
                    close_df
                    .reset_index()
                    .assign(
                        ano_mes    = lambda x: x["data"].dt.strftime("%Y%m"),
                        returns    = lambda x: x.groupby("ticker")["close"].pct_change(),
                        vol_neg_mm = lambda x: x["vol_negociado"] / 1000  # milhares → milhões
                    )
                    .dropna(subset=["returns"])
                    .merge(
                        mkt_cap_total[["ano_mes", "A_t"]],
                        on="ano_mes",
                        how="left"
                    )
                    .dropna(subset=["A_t"])
                )


                df_illiq_monthly = (
                    df_daily
                    .groupby(
                        ["ticker", "ano_mes"],
                        as_index=False
                    )
                    .apply(acharya_pedersen_illi)
                    .reset_index()
                    .rename(columns={None: "illiq_monthly"})
                    .sort_values(["ticker", "ano_mes"])
                )

                df_illiquidity = (
                    df_illiq_monthly
                    .assign(
                        liq_mm12 = lambda x: x.groupby("ticker")["illiq_monthly"].transform(
                            lambda y: y.rolling(12).mean()
                        )
                    )
                    .assign(
                        liq_mm12 = lambda x: x.groupby("ticker")["liq_mm12"].shift(1)
                    )
                    .dropna(subset=["liq_mm12"])
                    .assign(
                        ano_mes = lambda x: x["ano_mes"].astype(int)
                    )
                    [["ticker", "ano_mes", "liq_mm12"]]
                )


            except Exception as error:
                raise Exception(
                    f"Impossible to build liquidity indicators: {error}"
                    ) from error
            
            return df_illiquidity


        try:
            df_closes_trade_vol = (
                    sanitize_df(
                        (
                            pd.read_csv(get_config()["cotacoes_liquidez_data_feed"])
                            .set_axis(["ticker", "data", "close", "qt_negocios", "vol_negociado"], 
                                     axis="columns")
                        ),
                        cols_to_cast_float=["close", "vol_negociado"],
                        cols_to_cast_int=["qt_negocios"],
                        col_to_cast_date="data"
                    )
                    .sort_values(["ticker", "data"])
                    .dropna(subset="close")
                    .set_index("data")
            )

        except Exception as error:
            raise Exception(
                f"Impossible to read and sanitize closing and trade volume dataset: {error}"
                ) from error
        
        try:
            df_momentum = get_momentum_df(df_closes_trade_vol)
            df_illiquidity = get_liquidity_df(df_closes_trade_vol)
            df_bab = get_bab_df(df_closes_trade_vol)

            df_monthly_ind = (
                pd.merge(df_momentum, df_illiquidity, 
                         on=["ticker", "ano_mes"], how="inner")
                .merge(df_bab, on=["ticker", "ano_mes"], how="inner")
                .dropna(subset=["momentum_12_2", "beta_m24", "liq_mm12"])
                .drop("data", axis=1)
                .sort_values(["ano_mes", "ticker"])
                .rename(columns={"ano_mes": "data", "momentum_12_2": 
                                    "returns", "beta_m24": "beta", 
                                    "liq_mm12": "illiquidity"})
                .assign(
                    ano = lambda x: x["data"].astype(str).str.slice(0, 4)
                )
            )

            update_config("config_vars", "indicadores_mensais_file_path", "indicadores_mensais.csv")

            df_monthly_ind.to_csv(get_config()["indicadores_mensais_file_path"], index=False)
        
        except Exception as error:
            raise Exception(
                f"Impossible to build monthly indicators dataframe: {error}"
                ) from error

        return df_monthly_ind
    

    def transform_trimestral_indicators(self) -> pd.DataFrame:
        """
        Transforms trimestral indicators df
        
        :param self: instance
        :return: transformed dataframe
        :rtype: DataFrame
        """

        file_path = get_config()["indicadores_data_feed"]
        try:

            df_trimestral = (sanitize_df(
                
                    (pd.read_csv(file_path, encoding="latin")
                    .set_axis(["ticker", "tri", "market_cap", "preco_lucro", "lpa", "lucro", "qtd_acoes", 
                                   "qtd_acoes_ano", "patrimonio_liquido", "eqv_caixa", "data_balanco", "fin_cp", 
                                   "fin_lp", "close", "qt_negocios", "volume_negociado", "acionistas_discl_dt"], axis="columns"))
                ,
                cols_to_cast_float=["market_cap", 
                            "preco_lucro", "lpa", 
                            "lucro", "qtd_acoes",
                            "patrimonio_liquido",
                            "eqv_caixa", "fin_cp", "fin_lp",
                            "close", "qtd_acoes_ano"],
                cols_to_cast_int=[],
                col_to_cast_date=None
            )
            .assign(
                **{
                    "data": lambda x: x["tri"].str.slice(2),
                    "book-to-market-value": lambda x: x["patrimonio_liquido"] / x["market_cap"],
                    "ano" : lambda x: x["tri"].str.slice(2)
                    }
                )
            )
            

            update_config("config_vars", "indicadores_trimestrais_file_path", "indicadores_trimestrais.csv")

            df_trimestral.to_csv(get_config()["indicadores_trimestrais_file_path"])
    
        except Exception as error:
            raise Exception(f"Couldn't transform trimestral indicators dataset: {error}") from error


    def build_portfolios(self, indicator: Literal["market_cap", "book-to-market-value", 
                                                  "returns", "illiquidity", "beta"], 
                         indicator_analysis_period: Literal["monthly", "4T", "2T"],
                         ascending: bool=False):
        """
        Builds portfolios ordered by indicator
        
        :param self: instance param
        :param indicator: indicator name
        :type indicator: Literal["market_cap", "book-to-market-value", "returns", "illiquidity", "beta"]
        :param indicator_analysis_period: when should the indicator be analyzed
        :type indicator_analysis_period: Literal["monthly", "4T", "2T"]
        :param ascending: how to sort the portfolios
        """
        
        if indicator not in self.indicators:
            raise Exception(f"Input a valid indicator to build portfolios. Passed: {indicator}. Available: {self.indicators}")
        
        config = get_config()

        try:
            if indicator in ["returns", "illiquidity", "beta"]:
            
                file_path = config["indicadores_mensais_file_path"]

            else:

                file_path = config["indicadores_trimestrais_file_path"]
            
            df = (
                pd.read_csv(file_path)
            )

        except Exception as error:
            raise Exception(f"Error while reading the {indicator} file path {file_path}: {error}") from error

        portfolios_list = []

        periods = sorted(df["data"].unique())

        elegible_stocks = pd.read_csv(config["elegibilidade_final_file_path"])
        
        for i, data in enumerate(periods[1:]):
            
            i += 1

            ano = int(str(data)[:4])
            ano_mes_carteira = data
            
            # if ano == int(date.today().year) and indicator == "book-to-market-value":
            #     i -= 1

            if data == date.today().strftime("%Y%m") and indicator_analysis_period == "monthly":
                continue

            try:

                indicator_period = periods[i - 1] if indicator_analysis_period != "monthly" else periods[i]
                
                # importante: so podemos usar periods[i] aqui para os mensais pq os indicadores mensais já estão com lag (shift)
                # portanto, não alterar o shift nem isso aqui.

                df_year = df[df["data"] == indicator_period]

                if indicator_analysis_period != "monthly":
                    df_year = df_year[df_year["tri"] == f"{indicator_analysis_period}{indicator_period}"]

                if df_year.empty:
                    continue

                elegible_stocks_year = elegible_stocks[elegible_stocks["ano"] == ano]
                if elegible_stocks_year.empty:
                    continue

                if indicator == "book-to-market-value" or indicator == "market_cap":
                    df_year = df_year[(df_year[indicator] > 0) & (df_year[indicator] != np.inf)]

                df_year = df_year.drop_duplicates(subset="ticker")
                df_year = (
                    elegible_stocks_year[["ano", "ticker"]]
                    .merge(df_year, on="ticker", how="left")
                    .assign(
                        indicator_rank = lambda x: x[indicator].rank(method="first", ascending=ascending),
                        ano_mes_carteira = ano_mes_carteira,
                        tercile = lambda x: pd.qcut(x["indicator_rank"], 3, labels=[1, 2, 3]),
                    )
                    .sort_values(["data", "indicator_rank"])
                    .dropna(subset="tercile")
                    .assign(
                        data = lambda x: x["data"].astype(int)
                    )
                )[["ticker", "data", "ano_mes_carteira", indicator, "indicator_rank", "tercile"]]

                portfolios_list.append(df_year)
                
            except Exception as error:
                raise Exception(f"Couldn't build the portfolios for each period ({data}): {error}") from error
            
        portfolios_by_year = pd.concat(portfolios_list, ignore_index=True)

        portfolios_file_path = path.join(get_config()["config_vars"]["data_path"], 
                                         f"portfolios_by_year_{indicator}.csv")
        
        portfolios_by_year.to_csv(portfolios_file_path, index=False)

        return portfolios_by_year


    def build_factors(self, factor, file_path) -> pd.DataFrame:
        """
        Builds the risk factor time series by
        subtracting one tercile portfolio from another
        
        :param self: instance param
        :param factor: name of the risk factor to be built
        :param file_path: path of the indicators file to build the factor
        :return: None
        :rtype: pd.DataFrame
        """

        if factor not in self.factors:
            raise Exception(f"Input a valid indicator to build portfolios. Passed: {factor}. Available: {self.factors}")
        
        pattern = "%Y%m" if factor in ["momentum", "bab", "illiquidity"] else "%Y"
        
        config = get_config()
        returns = (
            sanitize_df(
                (
                    pd.read_csv(get_config()["cotacoes_liquidez_data_feed"])
                    .set_axis(["ticker", "data", "close", 
                               "qt_negs", "vol_negociado"], axis="columns")
                 ),
                 cols_to_cast_float=["close"],
                 cols_to_cast_int=[],
                 col_to_cast_date="data"
            )
            [["ticker", "data", "close"]]
            .dropna(subset="close")
            .assign(
                returns = lambda x: x.groupby("ticker")["close"].pct_change()
            )
            .dropna(subset="returns")
            .assign(
                ano_mes_carteira = lambda x: x["data"].dt.strftime(pattern).astype(int)
            )
        )

        try:
            df_indicators = (
                pd.read_csv(file_path)
            )
            
            # already in daily returns
            cdi_data = (
                get_bcb("cdi_diario", start="2009-01-01", end=datetime.today())
                .set_axis(["data", "rf_return"], axis="columns")
                .assign(
                    data = lambda x: pd.to_datetime(x["data"], format="%d/%m/%Y"),
                    rf_return = lambda x: x["rf_return"].astype(str).str.replace(",", ".").astype(float) / 100
                )
            )

            periods = sorted(df_indicators["ano_mes_carteira"].unique())
            factors_list = []

            indicator = list(df_indicators.columns)[-3]
            # cuidado para não quebrar

            for period in periods:

                df_factor = df_indicators[(df_indicators["ano_mes_carteira"] == period)]
                terciles_returns = []

                for tercile in sorted(df_factor["tercile"].unique()):

                    stocks = (
                        df_factor[df_factor["tercile"] == tercile]
                        ["ticker"].unique()
                    )

                    # trocado de soma / len pq se alguma acao nao teve retorno ela iria influenciar negativamente sem necessidade

                    returns_tercile = (
                        returns[(returns["ticker"].isin(stocks)) & 
                                (returns["ano_mes_carteira"] == period)]
                        .pivot(index="data", columns="ticker", values="returns")
                        .assign(
                            **{
                                f"retorno_carteira_tercil_{tercile}": lambda x: x[[col for col in x.columns]].mean(axis=1)
                            }
                        )
                    )[[f"retorno_carteira_tercil_{tercile}"]]

                    terciles_returns.append(returns_tercile)

                returns_terciles_df = (
                    pd.concat(terciles_returns, axis=1, ignore_index=True)
                    .set_axis(["retorno_carteira_tercil_1", 
                               "retorno_carteira_tercil_2", 
                               "retorno_carteira_tercil_3"], axis="columns")
                    .reset_index()
                )

                if factor == "bab":

                    beta_low = df_factor[df_factor["tercile"] == 1]["beta"].mean()
                    beta_high = df_factor[df_factor["tercile"] == 3]["beta"].mean()

                    returns_factor = (
                        returns_terciles_df
                        .merge(cdi_data, on="data", how="left")
                        .assign(
                            **{f"factor_{factor}": lambda x: ((x["retorno_carteira_tercil_1"] - x["rf_return"]) / beta_low) - 
                               ((x["retorno_carteira_tercil_3"] - x["rf_return"]) / beta_high)}
                        )
                    )

                else:
                    returns_factor = (
                        returns_terciles_df
                        .assign(
                            **{f"factor_{factor}": lambda x: x["retorno_carteira_tercil_1"] - 
                               x["retorno_carteira_tercil_3"]}
                        )
                    )

                factors_list.append(returns_factor)

            factor_time_series = pd.concat(factors_list, ignore_index=True)

            factor_path = path.join(get_config()["config_vars"]["data_path"], f"time_series_{factor}.csv")
            factor_time_series.to_csv(factor_path, index=False)
        
        except Exception as error:
            raise Exception(f"Couldn't build {factor} factor series.: {error}") from error
        
        return factor_time_series


    def build_market_factor(self):
        """
        builds the market factor
        
        :param self: instance param
        """

        try:

            elegible_stocks = (
                pd.read_csv(config["elegibilidade_final_file_path"])
                .sort_values(["ticker", "ano"])
            )[["ticker", "ano"]]

            df_mkt_cap = (
                        sanitize_df(
                            (pd.read_csv(config["liquidez_file_path"])
                            .set_axis(["ticker", "data", "liquidez", "market_cap"], axis="columns")
                            .drop("liquidez", axis=1)
                            .assign(
                                mes = lambda x: x["data"].str.slice(0, 3).map(self.months_dict),
                                ano = lambda x: x["data"].str.slice(4),
                                ano_mes = lambda x: x["ano"] + "-" + x["mes"]
                            )
                            [["ticker", "ano_mes", "market_cap"]]),
                    cols_to_cast_float=["market_cap"],
                    cols_to_cast_int=[],
                    col_to_cast_date="ano_mes",
                    date_formate_princip="%Y-%m"
                    )
                    .assign(
                        ano = lambda x: x["ano_mes"].dt.year,
                        ano_mes = lambda x: x["ano_mes"].dt.strftime("%Y%m"),
                        # shifting mkt_cap to avoid look ahead
                        market_cap = lambda x: x.groupby("ticker")["market_cap"].shift(1)
                    )
                    .dropna(subset="market_cap")[["ticker", "ano", "ano_mes", "market_cap"]]
                    .merge(
                    elegible_stocks, on=["ano", "ticker"],
                    how="right"
                )
            )

            df_returns = (
                    sanitize_df(
                        (
                            pd.read_csv(get_config()["cotacoes_liquidez_data_feed"])
                            .set_axis(["ticker", "data", "close", "qt_negocios", "vol_negociado"], 
                                     axis="columns")[["ticker", "data", "close"]]
                        ),
                        cols_to_cast_float=["close", "vol_negociado"],
                        cols_to_cast_int=["qt_negocios"],
                        col_to_cast_date="data"
                    )
                    .sort_values(["ticker", "data"])
                    .dropna(subset="close")
                    .assign(
                             ano_mes = lambda x: x["data"].dt.strftime("%Y%m"),
                             ano = lambda x: x["data"].dt.year,
                             stock_return = lambda x: x.groupby("ticker")["close"].pct_change()
                         )
                    .dropna(subset="stock_return")
            [["data", "ano", "ano_mes", "ticker", "stock_return"]]
            .merge(
                    elegible_stocks, on=["ano", "ticker"],
                    how="right"
                )
            )

            df_weights = (
                df_mkt_cap
                .assign(
                    weight = lambda x: (
                        x.groupby("ano_mes")["market_cap"]
                        .transform(lambda y: y / y.sum())
                    )
                )[["ticker", "ano_mes", "weight"]]
            )

            market_ret_df = (
                df_returns.merge(df_weights, on=["ticker", "ano_mes"], how="inner")
                .assign(mkt_return = lambda x: x["stock_return"] * x["weight"])
                .groupby("data", as_index=False)["mkt_return"].sum()
            )

            # already in daily returns
            cdi_data = (
                get_bcb("cdi_diario", start="2009-01-01", end=datetime.today())
                .set_axis(["data", "rf_return"], axis="columns")
                .assign(
                    data = lambda x: pd.to_datetime(x["data"], format="%d/%m/%Y"),
                    rf_return = lambda x: x["rf_return"].astype(str).str.replace(",", ".").astype(float) / 100
                )
            )

            returns_factor = (
                pd.merge(market_ret_df, cdi_data, on='data', how="left")
                .assign(
                    market_factor = lambda x: x["mkt_return"] - x["rf_return"]
                )
                .dropna(subset="market_factor")
                [["data", "market_factor"]]
            )

            factor_path = path.join(get_config()["config_vars"]["data_path"], 
                                    f"time_series_mkt_return.csv")
            
            returns_factor.to_csv(factor_path, index=False)

        except Exception as error:
            raise Exception(f"Couldn't build market factor: {error}") from error

        return returns_factor
    

    def __call__(self) -> None:
        """
        executes the hole routine
        
        :param self: instance param
        """

        # elegibility_df = self.get_elegibility()

        monthly_ind = self.get_monthly_indicators()

        self.transform_trimestral_indicators()

        hml_indicator = "book-to-market-value" # "try preco_lucro instead"
        hml_ascending = False if hml_indicator == "book-to-market-value" else True

        portfolios_by_year_size = self.build_portfolios("market_cap", "4T", ascending=True)
        portfolios_by_year_value = self.build_portfolios(hml_indicator, "2T", ascending=hml_ascending)
        portfolios_by_year_momentum = self.build_portfolios("returns", "monthly", ascending=False)
        portfolios_by_year_illiquidity = self.build_portfolios("illiquidity", "monthly", ascending=False)
        portfolios_by_year_beta = self.build_portfolios("beta", "monthly", ascending=True)
        
        market_cap_file_path = path.join(config["config_vars"]["data_path"], 
                                f"portfolios_by_year_market_cap.csv")
        
        btmv_file_path = path.join(config["config_vars"]["data_path"], 
                                f"portfolios_by_year_book-to-market-value.csv")
        
        returns_file_path = path.join(config["config_vars"]["data_path"], 
                                f"portfolios_by_year_returns.csv")

        illiquidity_file_path = path.join(config["config_vars"]["data_path"], 
                                f"portfolios_by_year_illiquidity.csv")
        
        beta_file_path = path.join(config["config_vars"]["data_path"], 
                                f"portfolios_by_year_beta.csv")
        
        all_factors_file_path = path.join(config["config_vars"]["data_path"], 
                                f"all_factors.csv")
        
        mkt_factor = self.build_market_factor()
        
        factor_size = self.build_factors("size", market_cap_file_path)
        factor_value = self.build_factors("value", btmv_file_path)
        factor_momentum = self.build_factors("momentum", returns_file_path)
        factor_illiquidity = self.build_factors("illiquidity", illiquidity_file_path)
        factor_bab = self.build_factors("bab", beta_file_path)

        factors = [mkt_factor, factor_size, factor_value,
                   factor_momentum, factor_illiquidity,
                   factor_bab]
        factors = [fac.drop([col for col in fac.columns if "retorno_carteira" in col], axis=1) 
                   for fac in factors]

        final_factors_base = reduce(lambda left, right: left.merge(right, on="data", how="left"), factors).dropna()

        final_factors_base.to_csv(f"{all_factors_file_path}", index=False)


if __name__ == "__main__":

    factor_builder = RiskFactors()
    factor_builder()