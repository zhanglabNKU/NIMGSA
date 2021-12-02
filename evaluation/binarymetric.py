import numpy as np
from sklearn.metrics import roc_curve

pos = 5430
neg = 495*383-5430
y1 = np.loadtxt('imcmda.txt',delimiter=',').flatten()
y2 = np.loadtxt('spm.txt',delimiter=',').flatten()
y3 = np.loadtxt('nimcgcn.txt',delimiter=',').flatten()
y4 = np.loadtxt('mclpmda.txt',delimiter=',').flatten()
y5 = np.loadtxt('gaemda.txt',delimiter=',').flatten()
y6 = np.loadtxt('nimgmda.txt',delimiter=',').flatten()
ldi = np.loadtxt('m-d.txt',delimiter=',').flatten()

def count(fpr,tpr,a):
    i = np.abs(fpr-a).argmin()
    fp = fpr[i] * neg
    tn = neg - fp
    tp = tpr[i] * pos
    fn = pos - tp
    return tp,tn,fp,fn

def binarymetrics(count):
    TP,TN,FP,FN = count
    Sn = TP/(TP+FN) # Sp=Rec
    Sp = TN/(TN+FP)
    Acc = (TN+TP)/(TN+TP+FN+FP)
    Pre = TP/(TP+FP)
    F1 = 2*(Pre*Sn)/(Pre+Sn)
    Mcc = (TP*TN-FP*FN)/np.sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))
    return Sp,Sn,Acc,Pre,F1,Mcc

def test(pred,a,msg):
    fpr,tpr,rocth = roc_curve(ldi,pred)
    Sp,Sn,Acc,Pre,F1,Mcc = binarymetrics(count(fpr,tpr,a))
    print('& {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(msg,Sn,Acc,Pre,F1,Mcc))

test(y1,0.05,'IMCMDA')
test(y2,0.05,'SPM')
test(y3,0.05,'NIMCGCN')
test(y4,0.05,'MCLPMDA')
test(y5,0.05,'GAEMDA')
test(y6,0.05,'NIMGSA')
test(y1,0.01,'IMCMDA')
test(y2,0.01,'SPM')
test(y3,0.01,'NIMCGCN')
test(y4,0.01,'MCLPMDA')
test(y5,0.01,'GAEMDA')
test(y6,0.01,'NIMGSA')