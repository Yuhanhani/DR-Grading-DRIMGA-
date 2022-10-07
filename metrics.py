#import scikit-learn
import skll # short for scikit-learn laboratory
import sklearn # short for scikit-learn
import imblearn

def quadratic_weighted_kappa(y_true, y_pred):
    kappa_score = skll.metrics.kappa(y_true, y_pred, weights='quadratic', allow_off_by_one=False)
    return kappa_score

def macro_auc(y_true, y_pred):
    macro_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, multi_class='ovo')  # change from ovr to ovo on 9th Aug.
    return macro_auc

def marco_precision(y_true, y_pred):
    macro_precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    return macro_precision

def marco_sensitivity(y_true, y_pred):   # same as recall (same definition)
    macro_sensitivity = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    return macro_sensitivity

def marco_specificity(y_true, y_pred):
    macro_specificity = imblearn.metrics.specificity_score(y_true, y_pred, average='macro')
    return macro_specificity
#-----------------------------------------------------------------------------------------------------------------------
def my_loss(y_true, y_pred):
    kappa_score = skll.metrics.kappa(y_true, y_pred, weights='quadratic', allow_off_by_one=False)
    loss = 1 - kappa_score
    return loss




