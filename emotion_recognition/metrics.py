"""Seven-class metrics; macro F1 and recall give every class equal weight."""
import numpy as np
from .data import CLASSES

def classification_metrics(labels,predictions):
    y=np.asarray(labels); p=np.asarray(predictions)
    if y.ndim!=1 or p.shape!=y.shape or not y.size or not np.isin(y,range(7)).all() or not np.isin(p,range(7)).all():
        raise ValueError("Labels/predictions must be nonempty equal 1D arrays with values 0..6.")
    matrix=np.zeros((7,7),dtype=np.int64)
    np.add.at(matrix,(y.astype(int),p.astype(int)),1)
    tp=matrix.diagonal(); support=matrix.sum(axis=1); predicted=matrix.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros(7,dtype=float),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros(7,dtype=float),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(7,dtype=float),where=precision+recall!=0)
    return {"accuracy":float(tp.sum()/y.size),"macro_f1":float(f1.mean()),"balanced_accuracy":float(recall.mean()),
            "sample_count":int(y.size),"labels":list(CLASSES),"confusion_matrix":matrix.tolist(),
            "per_class":[{"label":name,"precision":float(precision[index]),"recall":float(recall[index]),
                          "f1":float(f1[index]),"support":int(support[index])} for index,name in enumerate(CLASSES)],
            "confusion_order":"rows=true, columns=predicted; fixed FER label order",
            "zero_division":0}
