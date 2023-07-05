from sklearn import metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def get_evaluation(y_true, y_prob, list_metrics):
    # encoder = LabelEncoder()
    # encoder.classes_= np.load("./dataset/plcd/classes.npy")
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        # label_true = encoder.inverse_transform(y_true)
        # label_pred = encoder.inverse_transform(y_pred)
        label_true = y_true
        label_pred = y_pred
        output['F1'] = classification_report(label_true,label_pred)
        output['confusion_matrix'] = str(confusion_matrix(y_true, y_pred))
    return output