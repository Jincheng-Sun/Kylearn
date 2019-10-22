import matplotlib.pyplot as pyplot
pyplot.rcParams['savefig.dpi'] = 300  # pixel
pyplot.rcParams['figure.dpi'] = 300  # resolution
pyplot.rcParams["figure.figsize"] = [5,4] # figure size

def plot_pr_curve(recall, precision, auc, x_axis = 1, plt = pyplot):

    plt.plot(recall, precision, color="darkorange", label='Precision-Recall curve (area = %0.3f)' % auc)
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()