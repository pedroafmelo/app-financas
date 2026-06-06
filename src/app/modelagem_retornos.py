from __future__ import annotations
# -*- coding: UTF-8 -*-
"""
.py dedicado à construção da classe de modelagem dos retornos de um ativo ou carteira selecionada pelo usuário

"""

"""Import modules"""

from backtest import Backtest
from utils import get_config, get_bcb, sanitize_df
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from multiprocessing import cpu_count
from arch.univariate import ConstantMean, Normal, GARCH, arch_model
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import itertools
import random
from joblib import Parallel, delayed
from typing import Literal


class EconometricPredict:

    def __init__(self, JOBS: int = cpu_count() - 3, modelling_time: int = 6):

        self.JOBS = JOBS
        self.end = datetime.today()
        self.start = self.end - relativedelta(years=modelling_time)
        self.modelling_time = modelling_time

    def log_returns(prices: pd.Series | np.array, periodicity: str = "D"):

        prices = prices.resample(periodicity).last()
        log_return_series = np.log(prices / prices.shift()).dropna() * 100

        return log_return_series

    def model(self, series: pd.Series, 
              mean: Literal['Constant', 'Zero', 'LS', 
                            'AR', 'ARX', 'HAR', 'HARX'], 
              vol: Literal["ARCH", "GARCH", "EGARCH"],
              p: int, o: int, q: int, 
              dist: Literal['normal', 'gaussian', 't', 
                            'studentst', 'skewstudent', 
                            'skewt', 'ged', 'generalized error']):

        try:
            model = arch_model(
                series,
                mean = mean,
                vol = vol,
                p = p, 
                o = o, 
                q = q,
                dist=dist, 
                lags=5
            )

            result = model.fit(update_freq=0, disp="off")

        except Exception as error:
            print(f"Erro na construção do modelo {vol}: {error}")
            return 0

        return result

    def predict(self, model_result,
                method: Literal["simulation", "analytic", "bootstrap"],
                horizon: int, simulations: int | None = None) -> np.array | list:

        if method in ("simulation", "bootstrap") and simulations == None:
            raise ValueError(f"O método {method} precisa do valor de simulations")

        try:

            forecasts = model_result.forecast(start=date.today(),
                                              method=method,
                                              simulations=simulations,
                                              horizon=horizon)

            retornos = forecasts.mean
            variancia = forecasts.variance

            return retornos, variancia

        except Exception as error:
            print(f"Não foi possível realizar a previsão da vol e dos retornos: {error}")
            return [], []

    def optimize_model(self, series: pd.Series | np.array,
                       test_size: float = .3,              
                       vol: Literal["ARCH", "GARCH", "EGARCH"] = "GARCH"):

        tune_grid = list(
            itertools.product(
                *{
                    "p": range(1, 15),
                    "o": range(1, 12),
                    "q": range(1, 10),
                    "lags": range(0, 15),
                    "dist": ['normal', 'gaussian', 't', 
                            'studentst', 'skewstudent'],
                    "mean": ['Constant', 'Zero', 'LS', 
                            'AR', 'ARX', 'HAR', 'HARX']
                }.values()
            )
        )

        split_teste = round(len(series) * test_size)

        treino = series[:-split_teste]
        teste = series[-split_teste:]

        best_rmse = float("inf")
        best_params = {}
        metrics = pd.DataFrame()

        for p, o, q, lags, dist, mean in tune_grid:

            model = arch_model(treino, vol=vol,
                               p=p, o=o, q=q, lags=lags,
                               dist=dist, mean=mean)
            
            result = model.fit(update_freq=0, disp="off")

            pred = result.forecast(horizon=split_teste)

            rmse = root_mean_squared_error(teste, pred.mean.values[-1])

            metrics_temp = pd.DataFrame({
                "p": [p], "o": [o], "q": [q],
                "lags": [lags], "mean": [mean], 
                "dist": [dist], "rmse": [rmse],
                "pred_ret": pred.mean, "pred_var": pred.variance
            })

            metrics = pd.concat([metrics, metrics_temp])

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "p": p, "o": o, "q": q,
                    "lags": lags, "mean": mean, 
                    "dist": dist, "rmse": rmse
                }

        metrics = metrics.reset_index(drop=True).sort_values(by="rmse")

        return metrics, best_params, pred