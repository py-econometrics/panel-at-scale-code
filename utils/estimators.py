import pyfixest as pf
import numpy as np
from duckreg.estimators import DuckMundlak
import statsmodels.formula.api as smf

def twfe_fixest(df, T, T0):
    try:
        print("MemoryError: Not enough memory to run twfe_fixest.")
        m = pf.feols("Y~W | unit + time", df).tidy()
    except MemoryError:
        print("MemoryError: Not enough memory to run.")
    return None

def twfe_fixest_compressed(df, T, T0):
    try: 
        m = pf.feols(
            fml = "Y~W | unit + time",
            data = df, 
            use_compression = True, 
            reps = 1
        ).tidy()
    except MemoryError:
        print("MemoryError: Not enough memory to run.")

    return None

def twfe_statsmodels(df, T, T0):
    try: 
        m = smf.ols(formula="Y ~ W + C(unit) + C(time)", data=df).fit()
    except MemoryError:
        print("MemoryError: Not enough memory to run.")
    return None

def event_study_fixest(df, T, T0):
    try: 
        df["ever_treated"] = df.groupby("unit")["W"].transform("max")
        m = pf.feols(f"Y ~ i(time, ever_treated, ref = {T0-1}) | unit + time", df)
    except MemoryError:
        print("MemoryError: Not enough memory to run.")
    return None

def duck_mundlak(df, T, T0):

    mundlak = DuckMundlak(
        db_name="benchmarks.db",
        table_name="data",
        outcome_var="Y",
        covariates=["W"],
        unit_col="unit",
        time_col="time",
        cluster_col="unit",
        n_bootstraps=1,
        seed = 929
    )
    mundlak.fit()

    return mundlak