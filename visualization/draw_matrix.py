import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import seaborn as sb
import math

pyplot.rcParams['savefig.dpi'] = 300  # pixel
pyplot.rcParams['figure.dpi'] = 300  # resolution
pyplot.rcParams["figure.figsize"] = [0.5,0.5] # figure size

def draw_confusion_matrix(cm, labels, plt = pyplot, x_rotation=90, y_rotation=0, font_size=0.33, precision=False):


    if (precision):
        '''flip and rotate the confusion metrix'''
        labels = labels[::-1]
        cm = np.rot90(np.flip(cm, axis=0))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif (c == 0):
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)

    if (precision):
        cm.columns.name = 'True Label'
        cm.index.name = 'Predict Label'
    else:
        cm.index.name = 'True Label'
        cm.columns.name = 'Predict Label'

    sb.set(font_scale=font_size)

    sb.heatmap(cm, annot=annot, fmt='', cmap='Blues')
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)

    plt.show()


def plot_onedevice(proba, which, threshold, y_test, alarm_list):
    which_column = proba[:,which-1].copy()
    which_column[which_column >= threshold] = 1
    which_column[which_column < threshold] = 0

    # draw_confusion_matrix(cm, ['Normal', alarm_list[which - 1]], precision=True)