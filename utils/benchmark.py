import time 
import numpy as np
from utils.dgps import generate_benchmark_data
import pandas as pd

class Bench: 

    def __init__(self, N, T, T0, iter): 

        self.N = N
        self.T = T, 
        self.T0 = T0
        self.iter = iter
        # create pd.DataFrame & duckdb db
        self.df = generate_benchmark_data(N = N, T = T, T0 = T0)
        self.timings = {}

    def mark(self, fun, reps = 0):

        if fun in ["duckreg", "feols_compressed"]:
            fun_name = f"{fun.__name__} + reps = {reps}"
        else: 
            fun_name = fun.__name__
            
        self.timings[fun_name] = np.zeros(self.iter)

        for i in range(self.iter): 

            try:
                start = time.time()
                fun(df=self.df, T=self.T, T0=self.T0, reps = reps)
                self.timings[fun_name][i] = time.time() - start
            except MemoryError:
                print(f"MemoryError encountered in {fun_name}. Assigning np.nan.")
                self.timings[fun_name][i] = np.nan

    def to_dataframe(self): 

        self.timings_df = pd.DataFrame(self.timings)
        return self.timings_df
            
    def plot(self):
        pass