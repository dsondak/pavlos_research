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
    plt.show()

def viz_rl(pc,ac,idx,alp=0.7):
    """ plot the results of the experiment at [idx] in the pc (policies) and ac
    (accuracies) arrays """  
    accs = ac.mean(axis=0)
    me = len(pc[idx])
    cmap= plt.get_cmap('tab10')
    labels= ['conf','boundary','uniform','random','max_entropy']
    plt.figure(figsize=(10,8))
    plt.scatter(range(me),[10]*me,c=[cmap(1) if i=='conf' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.scatter(range(me),[10.1]*me,c=[cmap(1) if i=='boundary' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.scatter(range(me),[10.2]*me,c=[cmap(1) if i=='uniform' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.scatter(range(me),[10.3]*me,c=[cmap(1) if i=='random' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.scatter(range(me),[10.4]*me,c=[cmap(1) if i=='max_entropy' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.yticks([10,10.1,10.2,10.3,10.4], labels, rotation='horizontal')
    plt.plot(range(me),accs+10,alpha=alp)
    sns.despine()

def viz_rl_tl(pc,ac,idx,rwd=[],alp=0.7):
    """ plot the results of the experiment at [idx] in the pc (policies) and ac
    (accuracies) arrays """  
    #accs = ac.mean(axis=0)
    me = len(pc[idx])
    cmap= plt.get_cmap('tab10')
    labels= ['transfer','boundary']
    plt.figure(figsize=(10,8))
    plt.scatter(range(me),[10]*me,c=[cmap(1) if i=='transfer' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.scatter(range(me),[10.1]*me,c=[cmap(1) if i=='boundary' else cmap(0) for i in pc[idx]],alpha=alp)
    #plt.scatter(range(me),[10.2]*me,c=[cmap(1) if i=='uniform' else cmap(0) for i in pc[idx]],alpha=alp)
    #plt.scatter(range(me),[10.3]*me,c=[cmap(1) if i=='random' else cmap(0) for i in pc[idx]],alpha=alp)
    #plt.scatter(range(me),[10.4]*me,c=[cmap(1) if i=='max_entropy' else cmap(0) for i in pc[idx]],alpha=alp)
    plt.yticks([10,10.1], labels, rotation='horizontal')
    plt.plot(range(me),ac[idx,:]+10,alpha=alp)
    if rwd!=[]:
        plt.plot(range(me),rwd[idx,:]+10,alpha=alp)
    sns.despine()
