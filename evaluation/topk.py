import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,roc_auc_score,recall_score

pos = 5430
neg = 495*383-5430
y1 = np.loadtxt('imcmda.txt',delimiter=',')
y2 = np.loadtxt('spm.txt',delimiter=',')
y3 = np.loadtxt('nimcgcn.txt',delimiter=',')
y4 = np.loadtxt('mclpmda.txt',delimiter=',')
y5 = np.loadtxt('gaemda.txt',delimiter=',')
y6 = np.loadtxt('nimgmda.txt',delimiter=',')
ldi = np.loadtxt('m-d.txt',delimiter=',')

def topk(a,k):
    pred = []
    true = []
    for l in range(ldi.shape[0]):
        sort = np.argsort(-a[l,:])[:k]
        for d in sort:
            pred.append(a[l,d])
            true.append(ldi[l,d])
    
    return roc_auc_score(true,pred)

# def recall(a,k):
#     zero = np.zeros(ldi.shape)
#     for l in range(ldi.shape[0]):
#         sort = np.argsort(-a[l,:])[:k]
#         for d in sort:
#             zero[l,d] = 1.0
    
#     return 540*recall_score(ldi.flatten(),zero.flatten())

def recall(a,k):
    zero = np.zeros(ldi.shape)
    for d in range(ldi.shape[1]):
        sort = np.argsort(-a[:,d])[:k]
        for l in sort:
            zero[l,d] = 1.0
    
    return pos*recall_score(ldi.flatten(),zero.flatten())

def topktest(a):
    return np.array([
        recall(a,20),
        recall(a,40),
        recall(a,60),
        recall(a,80),
        recall(a,100)
    ])

def show(m,cc):
    c = cc.astype('int')
    print('{} & {} & {} & {} & {} & {} \\\\'.format(m,c[0],c[1],c[2],c[3],c[4]))

c1 = topktest(y1)
c2 = topktest(y2)
c3 = topktest(y3)
c4 = topktest(y4)
c5 = topktest(y5)
c6 = topktest(y6)

show('IMCMDA',c1)
show('SPM',c2)
show('NIMCGCN',c3)
show('MCLPMDA',c4)
show('GAEMDA',c5)
show('NIMGSA',c6)

ind = np.arange(0,10,2)
width = 0.2
fig,ax = plt.subplots(dpi=100)
rects6 = ax.bar(ind + 2.5*width, c6, width, label='NIMGSA')
rects5 = ax.bar(ind + 1.5*width, c5, width, label='GAEMDA')
rects4 = ax.bar(ind + 0.5*width, c4, width, label='MCLPMDA')
rects3 = ax.bar(ind - 0.5*width, c3, width, label='NIMCGCN')
rects2 = ax.bar(ind - 1.5*width, c2, width, label='SPM')
rects1 = ax.bar(ind - 2.5*width, c1, width, label='IMCMDA')

#ax.set_title('Scores by group and gender')
plt.xticks(ind,('Top 20', 'Top 40', 'Top 60', 'Top 80', 'Top 100'))
ax.legend(loc='center',bbox_to_anchor=(0.5,1.1),ncol=4)
plt.tight_layout()
#fig.savefig('recall1.eps')
fig.savefig('recall1.pdf')
plt.show()