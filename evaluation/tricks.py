import numpy as np
def predict_avoid_OOM(model, data, output_dim, *args):
    proba = np.empty([0, output_dim])
    for i in range(round(data.shape[0]/10000)+1):
        proba = np.concatenate([proba, model.get_proba(data[i*10000:i*10000+10000], *args)])
        print(i)
    print('Done')
    return proba


