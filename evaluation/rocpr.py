import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc

fs = 16

y1 = np.loadtxt('imcmda.txt',delimiter=',').flatten()
y2 = np.loadtxt('spm.txt',delimiter=',').flatten()
y3 = np.loadtxt('nimcgcn.txt',delimiter=',').flatten()
y4 = np.loadtxt('mclpmda.txt',delimiter=',').flatten()
y5 = np.loadtxt('gaemda.txt',delimiter=',').flatten()
y6 = np.loadtxt('nimgmda.txt',delimiter=',').flatten()
ytrue = np.loadtxt('m-d.txt',delimiter=',').flatten()

fpr1,tpr1,rocth1 = roc_curve(ytrue,y1)
fpr2,tpr2,rocth2 = roc_curve(ytrue,y2)
fpr3,tpr3,rocth3 = roc_curve(ytrue,y3)
fpr4,tpr4,rocth4 = roc_curve(ytrue,y4)
fpr5,tpr5,rocth5 = roc_curve(ytrue,y5)
fpr6,tpr6,rocth6 = roc_curve(ytrue,y6)

precision1,recall1,prth1 = precision_recall_curve(ytrue,y1)
precision2,recall2,prth2 = precision_recall_curve(ytrue,y2)
precision3,recall3,prth3 = precision_recall_curve(ytrue,y3)
precision4,recall4,prth4 = precision_recall_curve(ytrue,y4)
precision5,recall5,prth5 = precision_recall_curve(ytrue,y5)
precision6,recall6,prth6 = precision_recall_curve(ytrue,y6)

fig,ax = plt.subplots(1,2,figsize=(13,6),dpi=100)
plt.rcParams.update({"font.size":fs})

ax0 = ax[0]

ax0.plot(fpr6,tpr6,label='NIMGSA')
ax0.plot(fpr5,tpr5,label='GAEMDA')
ax0.plot(fpr4,tpr4,label='MCLPMDA')
ax0.plot(fpr3,tpr3,label='NIMCGCN')
ax0.plot(fpr2,tpr2,label='SPM')
ax0.plot(fpr1,tpr1,label='IMCMDA')
ax0.set_xlabel('False Positive Rate',fontsize=fs)
ax0.set_ylabel('True Positive Rate',fontsize=fs)
ax0.set_title('ROC Curves')
ax0.legend(loc='lower right')
ax0.tick_params(labelsize=fs)

ax1 = ax[1]

ax1.plot(recall6,precision6,label='NIMGSA')
ax1.plot(recall5,precision5,label='GAEMDA')
ax1.plot(recall4,precision4,label='MCLPMDA')
ax1.plot(recall3,precision3,label='NIMCGCN')
ax1.plot(recall2,precision2,label='SPM')
ax1.plot(recall1,precision1,label='IMCMDA')
ax1.set_xlabel('Recall',fontsize=fs)
ax1.set_ylabel('Precision',fontsize=fs)
ax1.set_title('PR Curves')
#ax1.legend(loc='upper right')
ax1.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig('auccc.pdf')
#plt.savefig('auc5cv2.eps')
plt.show()