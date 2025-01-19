import pandas as pd
import numpy as np
import math

df = pd.read_csv("/home/bethtian/fintech545/FinTech545_Spring2025/Projects/Project01/problem1.csv")
df_info = df.describe()
print(df_info)
arr = np.array(df['X'])
sorted_arr = np.sort(arr)
        
def calculateMoments(data):
    mean = np.mean(data)
    var = np.var(data)
    skewness = np.sum(((data-mean)/math.sqrt(var))**3)/len(data)
    kurtosis = np.sum(((data-mean)/math.sqrt(var))**4)/len(data) - 3
    return mean, var, skewness, kurtosis

if __name__ == "__main__":
    mean, variance, skewness, kurtosis = calculateMoments(arr)
    print(f"Mean: {mean:.4f}\nVariance: {variance:.4f}\nSkewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}")