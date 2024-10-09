import pyfixest as pf
import numpy as np
from duckreg.estimators import DuckMundlak, DuckMundlakEventStudy
import statsmodels.formula.api as smf

def twfe_fixest(df, T, T0, reps = 0):
    m = pf.feols("Y~W | unit + time", df, lean = True)
    return None

def twfe_fixest_compressed(df, T, T0, reps = 0):
    m = pf.feols(
        fml = "Y~W | unit + time",
        data = df, 
        use_compression = True, 
        reps = reps
    ).tidy()

    return None

def twfe_statsmodels(df, T, T0, reps = 0):
    m = smf.ols(formula="Y ~ W + C(unit) + C(time)", data=df).fit()
    return None

def event_study_statsmodels(df, T, T0, reps = 0):
    df['ever_treated'] = df.groupby('unit')['W'].transform('max')
    m = smf.ols(
        formula="Y ~ C(time):C(ever_treated) + C(unit) + C(time)", 
        data=df
    ).fit()  
    return m

def event_study_fixest(df, T, T0, reps = 0):
    df["ever_treated"] = df.groupby("unit")["W"].transform("max")
    m = pf.feols(f"Y ~ i(time, ever_treated, ref = {T0-1}) | unit + time", df, lean = True)
    return None

def duck_mundlak(df, T, T0, reps = 0):

    mundlak = DuckMundlak(
        db_name="benchmarks.db",
        table_name="data",
        outcome_var="Y",
        covariates=["W"],
        unit_col="unit",
        time_col="time",
        cluster_col="unit",
        n_bootstraps=reps,
        seed = 929
    )
    mundlak.fit()

    return mundlak

def duck_mundlak_event(df, T, T0, reps):

    mundlak = DuckMundlakEventStudy(
        db_name="benchmarks.db",
        table_name="data",
        outcome_var="Y",
        treatment_col="W",
        unit_col="unit",
        time_col="time",
        cluster_col="unit",
        n_bootstraps=0, # set to nonzero to get block-bootstrapped standard errors
        seed=42,
        pre_treat_interactions=True,
    )

    return mundlak