import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("assets/SARESP_train.csv")
columns = set(df.columns) - set(['nivel_profic_lp', 'nivel_profic_mat', 'nivel_profic_cie'])
df = df[columns]

import IPython; IPython.embed()
