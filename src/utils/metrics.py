from sklearn.metrics import precision_score, recall_score, f1_score

def compute_precision(labels=None, average='binary', **kwargs):
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred, labels=labels, average=average, **kwargs)
    return precision

def compute_recall(labels=None, average='binary', **kwargs):
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred, labels=labels, average=average, **kwargs)
    return recall

def compute_f1(labels=None, average='binary', **kwargs):
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, labels=labels, average=average, **kwargs)
    return f1