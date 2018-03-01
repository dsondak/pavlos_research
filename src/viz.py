import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(meta_epochs, accs, labs):
    """ Plot the results from the run of multiple policies sampling distributions""" 
    itrs = np.array(list(range(meta_epochs)))
    means, errs = [], []
    for ac in accs:
        means.append(np.mean(ac, axis=0))
        errs.append(np.std(ac,axis=0))

    plt.figure(figsize=(8,8))
    for ac_mean, ac_err, ac_lab in zip(means, errs, labs):
        plt.plot(itrs, ac_mean, label=ac_lab)
        plt.fill_between(itrs, ac_mean+ac_err, ac_mean-ac_err, alpha=0.4)

    plt.title('Accuracy of various active learning policies.')
    plt.xlabel('Meta Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    sns.despine()
