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
                timeout = 600  # 10 minutes in seconds
                
                # Run the function in a while loop to monitor its execution time
                while True:
                    if time.time() - start > timeout:
                        print(f"Timeout reached for {fun_name}. Assigning np.nan.")
                        self.timings[fun_name][i] = np.nan
                        break
                    
                    try:
                        fun(df=self.df, T=self.T, T0=self.T0, reps=reps)
                        self.timings[fun_name][i] = time.time() - start
                        break  # Break if the function completes within the time limit
                    
                    except MemoryError:
                        print(f"MemoryError encountered in {fun_name}. Assigning np.nan.")
                        self.timings[fun_name][i] = np.nan
                        break

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                self.timings[fun_name][i] = np.nan

        self.timings_df = pd.DataFrame(self.timings)

        return self.timings_df

    def to_dataframe(self): 

        self.timings_df = pd.DataFrame(self.timings)
        return self.timings_df
            
    def plot(self):
        pass