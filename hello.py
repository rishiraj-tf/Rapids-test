print("hello")
print(1+2)

import numpy as np
import pandas as pd

# 1) NumPy array basics
arr = np.arange(1, 11)               # [1, 2, …, 10]
print("NumPy array:", arr)
print("  sum =", arr.sum(), ", mean =", arr.mean())

# 2) Pandas Series operations
ser = pd.Series(arr, name="values")
print("\nPandas Series:")
print(ser.head())
print("  describe():\n", ser.describe())

# 3) Pandas DataFrame basics
df = pd.DataFrame({
    "x": arr,
    "y": arr * 2,
    "z": np.random.rand(10)
})
print("\nDataFrame:\n", df)


import cudf
df = cudf.DataFrame({
    "x": arr,
    "y": arr * 2,
    "z": np.random.rand(10)
})
print("\nDataFrame:\n", df)