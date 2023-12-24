from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_clf(pred_fn, feat, X_, Y_, threshold=0.5):
    preds = pred_fn(X_)
    mse = mean_squared_error(Y_, preds)
    r2 = r2_score(Y_, preds)
    ret = {'MSE': mse, 'R2': r2}
    if feat.endswith('_binary'):
        mat = confusion_matrix(Y_, preds > threshold)
        try:
            ret['TN'] = mat[0, 0]
            ret['FP'] = mat[0, 1]
            ret['FN'] = mat[1, 0]
            ret['TP'] = mat[1, 1]
            #  (1-FNR, hitrate, recall)
            ret['TPR'] = ret['TP'] / (ret['TP'] + ret['FN'])
            ret['TNR'] = ret['TN'] / (ret['FP'] + ret['TN'])
            ret['FN/FP'] = ret['FN'] / ret['FP']
            ret['Accuracy'] = (ret['TP'] + ret['TN']) / (ret['TP'] + ret['FN'] + ret['FP'] + ret['TN'])
        except IndexError:
            print(mat.shape)
    ret['Threshold'] = threshold
    return ret
