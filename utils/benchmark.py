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

    def mark(self, fun):

        fun_name = fun.__name__  # Get the name of the function
        self.timings[fun_name] = np.zeros(self.iter)

        for i in range(self.iter): 
            start = time.time()
            fun(df = self.df, T = self.T, T0 = self.T0)
            self.timings[fun_name][i] = time.time() - start

    def to_dataframe(self): 

        self.timings_df = pd.DataFrame(self.timings)
        return self.timings_df
            
    def plot(self):
        pass