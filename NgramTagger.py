# Loading Libraries 
from music21 import *
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger, NgramTagger
from nltk.corpus import treebank
import nltk
import dataset
import numpy as np
import random
import joblib
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def split_data(train_sents, ratio=(.80,.10,.10), verbose=True):
    if sum([s*100 for s in ratio]) != 100:
        raise Exception('Split ratios should sum to one')
    print('----[INFO] Splitting dataset') if verbose else None
    l_train = int(np.round(len(train_sents)*ratio[0]))
    l_test = int(np.round(len(train_sents)*ratio[1]))
    l_dev = int(np.round(len(train_sents)*ratio[2]))
    # For now, train each label separately
    data_notes = [[(x[0],x[1]) for x in sent] for sent in train_sents]    
    data_dur = [[(x[0],x[2]) for x in sent] for sent in train_sents]    
    
    XY_train_n = data_notes[:l_train]
    XY_train_t = data_dur[:l_train]
    
    XY_dev_n = data_notes[l_train:l_train+l_dev]
    XY_dev_t = data_dur[l_train:l_train+l_dev]
    
    XY_test_n = data_notes[l_train+l_dev:]
    XY_test_t = data_dur[l_train+l_dev:]
    
    print('----[INFO] Done') if verbose else None
    return ((XY_train_n,XY_train_t), (XY_dev_n,XY_dev_t), (XY_test_n, XY_test_t))

def train_tagger(XY_train,_type, cutoff=(0,0), save=False, verbose=True):
    (c1,c2)=cutoff
    print('----[INFO] Training models') if verbose else None
    # Initializing taggers and backoffs
    n1 = UnigramTagger(XY_train[0],cutoff=c1) 
    t1 = UnigramTagger(XY_train[1],cutoff=c2) 
    n2 = BigramTagger(XY_train[0],backoff=n1,cutoff=c1) 
    t2 = BigramTagger(XY_train[1],backoff=t1,cutoff=c2) 
    n3 = TrigramTagger(XY_train[0],backoff=n2,cutoff=c1) 
    t3 = TrigramTagger(XY_train[1],backoff=t2,cutoff=c2) 
    n4 = NgramTagger(4, XY_train[0],backoff=n3,cutoff=c1) 
    t4 = NgramTagger(4, XY_train[1],backoff=t3,cutoff=c2)
    
    if save==True:
        save_models([n1,t1,n2,t2,n3,t3,n4,t4], _type)
    print('----[INFO] Done') if verbose else None
    return [(n1,t1),(n2,t2),(n3,t3),(n4,t4)]

def predict(model_notes,model_time, sent, verbose=False): # Remember, one sentence without the labels
    print('----[INFO] Computing predictions') if verbose else None
    notes = model_notes.tag(sent)
    dur = model_time.tag(sent)
    
    print('----[INFO] Processing and cleaning predictions') if verbose else None
    #Postprocessing predictions to account for OOV words
    if notes[0][1] == None:
        # If we start None, not much we can do, so start C
        notes[0] = (notes[0][0], 'C2')
    
    if dur[0][1] == None:
        # If we start None, not much we can do, so start with 75ms
        dur[0] = (dur[0][0], '75')
        
    prev_n = notes[0] 
    prev_d = dur[0]
    i=0
    for n,d in zip(notes,dur):
        if n[1] == None:
            # Choosing a ratio for now
            ratio = [2,3,5]
            f_prev = dataset.note2freq(prev_n[1])
            f = f_prev * ratio[random.randrange(0,len(ratio))]
            notes[i] = (notes[i][0],dataset.pitch(f))
            
        if d[1] == None:
            dur[i] = (dur[i][0], prev_d[1])#(dur[i][0],all_dur[r])
            
        prev_n=notes[i]
        prev_d=dur[i]
        i=i+1
        
    pred_n = [n[1] for n in notes]
    pred_t = [d[1] for d in dur]
    
    print('----[INFO] Done') if verbose else None
    return (pred_n, pred_t)
    
def CMatrix(gold, prediction, verbose=False): # Both tuples for n and t
    print('----[INFO] Computing confusion matrix') if verbose else None
    cm_n = confusion_matrix(gold[0], prediction[0])
    cm_d = confusion_matrix(gold[1], prediction[1])
    # print(cm_n.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    print('----[INFO] Done') if verbose else None
    return (cm_n,cm_d)

def save_models(models,_type, verbose=True):
    # Saving model 
    j=1
    print('----[INFO] Saving models') if verbose else None
    for i in range(0,len(models),2):    
        joblib.dump(models[i], str(j)+'gram_n_tagger_'+_type+'.pkl') 
        joblib.dump(models[i+1], str(j)+'gram_t_tagger_'+_type+'.pkl') 
        j=j+1
    print('----[INFO] Done') if verbose else None

def keyStream(data,M,stepSize): # M: Window Size; stepSize: stride 
    N = len(data)
    notes = [d[0] for d in data]
    del_t = [d[1] for d in data]
    # Assume tempo constant and is at 60 bpm
    music_notes = stream.Stream()
    # Getting total notes
    for n,d in zip(notes[:N], del_t[:N]):
        music_notes.append(note.Note(n))
    
    keys = []
    # Rolling window of size M with step size 5 (i.e. stride)
    for i in range(0,len(music_notes),stepSize):
        w_notes = music_notes[i:i+M]
        keys = keys + [w_notes.analyze('key')]
    
    return keys

def compute_metrics(yn_pred, yt_pred, yn_true, yt_true):    
    n_metric = precision_recall_fscore_support(yn_true,yn_pred,average='weighted',
                                           zero_division=0)
    t_metric = precision_recall_fscore_support(yt_true,yt_pred,average='weighted',
                                           zero_division=0)
    
    return (n_metric, t_metric)

def conventional_metrics(X_test,Y_test_n,Y_test_t,mn,mt, verbose=False):
    score = 0
    metrics_n = []
    metrics_t = []
    accuracy = []
    cm = []
    for sent_x,sent_yn,sent_yt in zip(X_test,Y_test_n,Y_test_t):
        y_test = (sent_yn,sent_yt)
        prediction = predict(mn,mt,sent_x)
        cm = cm + [CMatrix(y_test, prediction)]
        metrics = compute_metrics(*prediction,sent_yn,sent_yt)
        accuracy = accuracy + [(accuracy_score(sent_yn,prediction[0]), accuracy_score(sent_yt,prediction[1]))] 
        metrics_n = metrics_n + [metrics[0]]
        metrics_t = metrics_t + [metrics[1]]
    
    prec_n = sum([m[0] for m in metrics_n])/len(metrics_n)
    prec_t = sum([m[0] for m in metrics_t])/len(metrics_t)
    
    rec_n = sum([m[1] for m in metrics_n])/len(metrics_n)
    rec_t = sum([m[1] for m in metrics_t])/len(metrics_t)
        
    f_n = sum([m[2] for m in metrics_n])/len(metrics_n)
    f_t = sum([m[2] for m in metrics_t])/len(metrics_t)
    
    acc_n = sum([a[0] for a in accuracy])/len(metrics_n)
    acc_t = sum([a[1] for a in accuracy])/len(metrics_t)

    print('Precision for Notes and Dur: ' + str(prec_n) + ',' + str(prec_t)) if verbose else None
    print('Recall for Notes and Dur: ' + str(acc_n) + ',' + str(acc_t)) if verbose else None
    print('Accuracy for Notes and Dur: ' + str(acc_n) + ',' + str(acc_t)) if verbose else None
    print('F1-Score for Notes and Dur: ' + str(f_n) + ',' + str(f_t)) if verbose else None
    
    return ((prec_n,prec_t), (rec_n,rec_t), (f_n,f_t), (acc_n,acc_t))
    
def key_similarity(X_test,Y_test_n,mn,mt):
    evals = []    
    #print('----[INFO] Computing accuracy')
    print('----[INFO] Computing streams of keys')
    for sent_x, sent_y in zip(X_test,Y_test_n): # Iterating over the ground truth for each song
        keys_gt = keyStream(sent_y,10,1) # Ground truth
        keys_pr = keyStream(predict(mn,mt,sent_x)[0],10,1) # Prediction
        # Getting number of 'hits' vs 'misses'
        b = [k_t==k_p for k_t,k_p in zip(keys_gt,keys_pr)]
        evals = evals + [sum(b)/len(b)]
        
    return sum(evals)/len(evals)

def grid_search(song_data, tune_iter=None, ratio=(.80,.10,.10), par=0, plot=True, verbose=True, save=True):
    print('[INFO] Computing grid search...' ) if verbose else None
    # par = [N, cutoff]
    #train_sents = dataset.sentences
    (sent_train, sent_dev, sent_test) = split_data(song_data, ratio=ratio)
    
    X_test = [[t[0] for t in data] for data in sent_test[0]]
    Y_test_n = [[t[1] for t in data] for data in sent_test[0]]
    Y_test_t = [[t[1] for t in data] for data in sent_test[1]]
    
    # For plotting performance curve
    X_dev = [[t[0] for t in data] for data in sent_dev[0]]
    Y_dev_n = [[t[1] for t in data] for data in sent_dev[0]]
    Y_dev_t = [[t[1] for t in data] for data in sent_dev[1]]
    
    X_train = [[t[0] for t in data] for data in sent_train[0]]
    Y_train_n = [[t[1] for t in data] for data in sent_train[0]]
    Y_train_t = [[t[1] for t in data] for data in sent_train[1]]    
    
    i=0
    accs_dev = []
    accs_train = []
    prec_dev = []
    prec_train = []
    rec_dev = []
    rec_train = []        
    f1_dev = []
    f1_train = []
    columns = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    metrics_n_dev = pd.DataFrame(0, index=[t[0] for t in tune_iter], columns=columns)
    metrics_n_train = pd.DataFrame(0, index=[t[0] for t in tune_iter], columns=columns)
    metrics_t_dev = pd.DataFrame(0, index=[t[1] for t in tune_iter], columns=columns)
    metrics_t_train = pd.DataFrame(0, index=[t[1] for t in tune_iter], columns=columns)
    # Evaluating the 4gram model for each cutoff value
    print('Cutoff \t Precision \t Recall \t F1-Score \t Accuracy')
    grid_n = np.zeros((4,4,len(tune_iter))) # Grid to store performance values for each model and cutoffs
    grid_t = np.zeros((4,4,len(tune_iter))) # Grid to store performance values for each model and cutoffs
    model_names=['UnigramTagger', 'BigramTagger', 'TrigramTagger', 'NgramTagger']
    for j in range(0,4): # Iterating over the four NgramTagger
        for it in tune_iter:
            (c1,c2) = it
            model = train_tagger(sent_train,'beatSim', cutoff=it, save=False, verbose=False)
            (mn,mt) = (model[j][0],model[j][1])    
            performance_conv_dev = conventional_metrics(X_dev, Y_dev_n, Y_dev_t,mn,mt)
            performance_conv_train = conventional_metrics(X_train, Y_train_n, Y_train_t,mn,mt)
            # key_acc = key_similarity(X_test, Y_test_n,mn,mt)
            # print('Key similarity: ' + str(key_acc))
            prec_dev = prec_dev + [performance_conv_dev[0]]
            prec_train = prec_train + [performance_conv_train[0]]
            
            rec_dev = rec_dev + [performance_conv_dev[1]]
            rec_train = rec_train + [performance_conv_train[1]]
            
            accs_dev = accs_dev + [performance_conv_dev[2]]
            accs_train = accs_train + [performance_conv_train[2]]
            
            f1_dev = f1_dev + [performance_conv_dev[3]]
            f1_train = f1_train + [performance_conv_train[3]]
            
            print(model_names[j] + str(it) + '\t' + str(prec_dev[i]) + '\t' + str(rec_dev[i]) + '\t' + str(f1_dev[i]) + '\t' + str(accs_dev[i]))
            metrics_n_dev.loc[c1,columns] = [prec_dev[i][0], rec_dev[i][0], f1_dev[i][0], accs_dev[i][0]]
            metrics_t_dev.loc[c2,columns] = [prec_dev[i][1], rec_dev[i][1], f1_dev[i][1], accs_dev[i][1]]
            metrics_n_train.loc[c1,columns] = [prec_train[i][0], rec_train[i][0], f1_train[i][0], accs_train[i][0]]
            metrics_t_train.loc[c2,columns] = [prec_train[i][1], rec_train[i][1], f1_train[i][1], accs_train[i][1]]
            i=i+1
        
        for k in range(0,4): # Iterating over the four metrics: precision, recall f1-score and accuracy 
            grid_n[k,j,:] = metrics_n_dev.iloc[:,k].values.T
            grid_t[k,j,:] = metrics_t_dev.iloc[:,k].values.T

    
    
    stds_n = metrics_n_dev.std().values
    stds_t = metrics_t_dev.std().values
    # For now, i'll use the metric with the highest variance
    MOI = stds_n.argmax() if max(stds_n) > max(stds_t) else stds_t.argmax()
    
    
    max_n = grid_n[MOI].max(axis=1)
    max_t = grid_t[MOI].max(axis=1)
    N_n = np.where((max_n/np.max(max_n))==1.0)[0][-1]
    N_t = np.where((max_t/np.max(max_t))==1.0)[0][-1]
    
    i_n_max = np.unravel_index(grid_n[MOI].argmax(),grid_n[MOI].shape)
    i_t_max = np.unravel_index(grid_t[MOI].argmax(),grid_t[MOI].shape)   
    
    optimal_hyperpar_n = model[N_n][0]
    optimal_hyperpar_t = model[N_t][1]
    
    print('Best NgramTagger is ' + str((optimal_hyperpar_n, optimal_hyperpar_t)))
    print('Best Cutoff is ' + str((i_n_max[1], i_t_max[1])))
    
    print('[INFO] Plotting results') if verbose else None
    if plot:
        # Plotting the grid search results if par==1
        (fig_n, axes_n) = plt.subplots(2,2)
        (fig_t, axes_t) = plt.subplots(2,2)
        #fig_n.set_size_inches(20, 7)
        #fig_t.set_size_inches(20, 7)
        for i,(grid,fig,axes) in enumerate(zip([grid_n, grid_t], [fig_n, fig_t], [axes_n, axes_t])):
            for j, (ax,grid) in enumerate(zip(axes.reshape(-1),grid)):
                ax.set_title(columns[j])
                xticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x))
                yticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x+1))
                ax.yaxis.set_major_formatter(yticks)
                ax.xaxis.set_major_formatter(xticks)
                im = ax.imshow(grid*100, cmap=plt.cm.gist_heat)
                ax.set_xlabel('Cutoffs')
                ax.set_ylabel('N')
            fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cax, label='Performance (%)')
            fig.suptitle(['Notes', 'Durations'][i], fontsize=25, fontweight='book', 
            verticalalignment='baseline')
            #fig.tight_layout()
              
    
    print('[INFO] Finding the best model...') if verbose else None
    stds_n = metrics_n_dev.std().values
    stds_t = metrics_t_dev.std().values

    # For now, i'll use the metric with the highest variance
    MOI = stds_n.argmax() if max(stds_n) > max(stds_t) else stds_t.argmax()
        
    print('[INFO] Hyperparameter selection based on ' + columns[MOI]) if verbose else None
    
    print('[INFO] Re-training model with optimal hyperparameters...' ) if verbose else None
    model = train_tagger(sent_train,'beatSim', cutoff=(i_n_max[1], i_t_max[1]), save=save)
    model = (model[N_n][0], model[N_t][1])
    print(model)
    print('[INFO] Done') if verbose else None
    return ((metrics_n_dev, metrics_t_dev), (metrics_n_train, metrics_t_train)), (grid_n, grid_t), model
    
def tune_model(song_data, ratio=(.80,.10,.10), par=0, plot=True, verbose=True, save=True):
    print('[INFO] Computing tuning curve...' ) if verbose else None
    # par = [N, cutoff]
    #train_sents = dataset.sentences
    (sent_train, sent_dev, sent_test) = split_data(song_data, ratio=ratio)
    
    X_test = [[t[0] for t in data] for data in sent_test[0]]
    Y_test_n = [[t[1] for t in data] for data in sent_test[0]]
    Y_test_t = [[t[1] for t in data] for data in sent_test[1]]
    
    # For plotting performance curve
    X_dev = [[t[0] for t in data] for data in sent_dev[0]]
    Y_dev_n = [[t[1] for t in data] for data in sent_dev[0]]
    Y_dev_t = [[t[1] for t in data] for data in sent_dev[1]]
    
    X_train = [[t[0] for t in data] for data in sent_train[0]]
    Y_train_n = [[t[1] for t in data] for data in sent_train[0]]
    Y_train_t = [[t[1] for t in data] for data in sent_train[1]]
    
    model = train_tagger(sent_train,'beatSim',save=False)
    i=0
    accs_dev = []
    accs_train = []
    prec_dev = []
    prec_train = []
    rec_dev = []
    rec_train = []        
    f1_dev = []
    f1_train = []
    columns = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    index = ['UnigramTagger', 'BigramTagger', 'TrigramTagger', 'NgramTagger']
    metrics_n_dev = pd.DataFrame(0, index=index, columns=columns)
    metrics_n_train = pd.DataFrame(0, index=index, columns=columns)
    metrics_t_dev = pd.DataFrame(0, index=index, columns=columns)
    metrics_t_train = pd.DataFrame(0, index=index, columns=columns)
    # Evaluating the 4gram model for each cutoff value
    for m in model:
        (mn,mt) = (m[0],m[1])
        performance_conv_dev = conventional_metrics(X_dev, Y_dev_n, Y_dev_t,mn,mt)
        performance_conv_train = conventional_metrics(X_train, Y_train_n, Y_train_t,mn,mt)
        # key_acc = key_similarity(X_test, Y_test_n,mn,mt)
        # print('Key similarity: ' + str(key_acc))
        prec_dev = prec_dev + [performance_conv_dev[0]]
        prec_train = prec_train + [performance_conv_train[0]]
        
        rec_dev = rec_dev + [performance_conv_dev[1]]
        rec_train = rec_train + [performance_conv_train[1]]
        
        accs_dev = accs_dev + [performance_conv_dev[2]]
        accs_train = accs_train + [performance_conv_train[2]]
        
        f1_dev = f1_dev + [performance_conv_dev[3]]
        f1_train = f1_train + [performance_conv_train[3]]
        
        metrics_n_dev.loc[index[i],columns] = [prec_dev[i][0], rec_dev[i][0], f1_dev[i][0], accs_dev[i][0]]
        metrics_t_dev.loc[index[i],columns] = [prec_dev[i][1], rec_dev[i][1], f1_dev[i][1], accs_dev[i][1]]
        metrics_n_train.loc[index[i],columns] = [prec_train[i][0], rec_train[i][0], f1_train[i][0], accs_train[i][0]]
        metrics_t_train.loc[index[i],columns] = [prec_train[i][1], rec_train[i][1], f1_train[i][1], accs_train[i][1]]
        i=i+1                        
                   
    print('[INFO] Plotting results') if verbose else None
    if plot:  
        # Plotting tuning curves
        (fig, ax) = plt.subplots(2,1)
        fig.set_size_inches(18.5, 10.5*2)
        yticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*100))
        ax[0].yaxis.set_major_formatter(yticks)
        ax[1].yaxis.set_major_formatter(yticks)
        
        styles=['r','b','g','m']
        # Plotting for notes
        metrics_n_dev.reset_index().plot(x='index',y=columns, style=styles,label=['Dev ' + c for c in columns], ax=ax[0], title='Notes')
        metrics_n_train.reset_index().plot(x='index',y=columns, style=[s+'--' for s in styles],
                                           label=['Training ' + c for c in columns], ax=ax[0], title='Notes')
        
        # Plotting for durations
        metrics_t_dev.reset_index().plot(x='index',y=columns, style=styles,label=['Dev ' + c for c in columns], 
                                         ax=ax[1], title='Durations')
        metrics_t_train.reset_index().plot(x='index',y=columns, style=[s+'--' for s in styles],
                                           label=['Training ' + c for c in columns], ax=ax[1], title='Durations')
        
        for a in ax:
            a.legend(loc=2, prop={'size': 10})
            a.set_xlabel([None, 'Cutoff'][par])
            a.set_ylabel('Performance (%)')
            a.grid(axis='both')
        
    # Returning the best model with its performance
    print('[INFO] Finding the best model...') if verbose else None
    i_n = [np.argmax(list(metrics_n_dev.loc[:,'Precision'])), np.argmax(list(metrics_n_dev.loc[:,'Recall'])), 
           np.argmax(list(metrics_n_dev.loc[:,'F1-Score'])), np.argmax(list(metrics_n_dev.loc[:,'Accuracy']))]

    i_t = [np.argmax(list(metrics_t_dev.loc[:,'Precision'])), np.argmax(list(metrics_t_dev.loc[:,'Recall'])), 
           np.argmax(list(metrics_t_dev.loc[:,'F1-Score'])), np.argmax(list(metrics_t_dev.loc[:,'Accuracy']))]

    stds_n = metrics_n_dev.std().values
    stds_t = metrics_t_dev.std().values
    # For now, i'll use the metric with the highest variance
    MOI = stds_n.argmax() if max(stds_n) > max(stds_t) else stds_t.argmax()
        
    print('[INFO] Hyperparameter selection based on ' + columns[MOI]) if verbose else None
    # Getting optimal hyperparameter based on the metric chosen
    optimal_hyperpar_n = model[i_n[MOI]][0]
    optimal_hyperpar_t = model[i_t[MOI]][1]
    print('Best NgramTagger is ' + str((optimal_hyperpar_n, optimal_hyperpar_t))) if verbose else None

    print('[INFO] Done') if verbose else None
    return ((metrics_n_dev, metrics_t_dev), (metrics_n_train, metrics_t_train)), model


def __main__():
    (words,notes,del_t,vocab,song_data) = dataset.get_data()
    (sent_train, sent_dev, sent_test) = split_data(song_data, ratio=(.80,0.10,0.10))
    
    X_test = [[t[0] for t in data] for data in sent_test[0]]
    Y_test_n = [[t[1] for t in data] for data in sent_test[0]]
    Y_test_t = [[t[1] for t in data] for data in sent_test[1]]
    
    # Tuning Ngram
    metrics_1 = tune_model(song_data, par=0, plot=True, verbose=True)
    
    # Tuning cutoff value
    cutoff_values = [(i,i) for i in range(0,10,1)]
    (metrics_2,grid,model) = grid_search(song_data, tune_iter=cutoff_values, par=1, plot=True, verbose=True)
    
    print('[INFO] Computing test set metrics')
    test_metrics = conventional_metrics(X_test,Y_test_n,Y_test_t,model[0],model[1], verbose=True)
        #key_acc = key_similarity(X_test, Y_test_n,m[0],m[1])
        #print('Key similarity: ' + str(key_acc))
        #print('\n')
    
    # Tuning doesn't seem to improve performance, so we will use cutoff of zero
    train_tagger(sent_train,'beatSim', cutoff=(0, 0), save=True)
        
if __name__ == "__main__":
    print('[INFO] Starting training procedure')
    __main__()
    print('[INFO] Done')

    
