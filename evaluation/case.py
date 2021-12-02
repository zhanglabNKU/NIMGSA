import numpy as np
import pandas as pd

y = np.loadtxt('nimgmda.txt',delimiter=',')
x = np.loadtxt('m-d.txt',delimiter=',')
r = y - x
lnc = pd.read_csv('miRNA name.csv')
d = pd.read_csv('disease name.csv')
df = pd.DataFrame(columns=('disease name','miRNA name','score'))
li,di = np.where(r>0.1)
for i in range(len(li)):
    df=df.append({'disease name':d.loc[di[i],'name'], 'miRNA name':lnc.loc[li[i],'name'], 'score':r[li[i],di[i]]}, ignore_index=True)

df=df.sort_values(by='score',ascending=False)
df.to_csv('prediction.csv')