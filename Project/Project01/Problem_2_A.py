import pandas as pd


df = pd.read_csv("/home/bethtian/fintech545/FinTech545_Spring2025/Projects/Project01/problem2.csv")
def main():
    cov_matrix = df.cov()
    cov_matrix.to_excel("/home/bethtian/fintech545/Project/Project01/Problem_2_A.xlsx")
    return 0

if __name__ == "__main__":
    main()