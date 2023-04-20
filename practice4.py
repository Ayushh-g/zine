import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ECG_training (org).csv",chunksize=3)
for chunk in data:
    # data=chunk[0].str.split(",",expand=True)
    X=chunk['ECG'].str.split(',',expand=True)
    y=chunk['Classification']
    X=np.array(X,dtype=float)
    X_A=X[y=='A']
    X_N=X[y=='N']
    plt.plot(X_A.T)
    plt.title('class a')
    plt.show()
    plt.plot(X_N.T)
    plt.title('class n')
    plt.show()
    break
