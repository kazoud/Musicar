import dataset
import random
import joblib
import os
import play_song
import wavio

min_pent = [30,36,40,45,54,60]
maj_pent = [24,27,30,36,40,48]
egyp_susp = [24,27,32,36,42,48]
bl_minor = [15,18,20,24,27,30] 
bl_major = [24,27,32,36,40,48]

all_notes = list(set(dataset.notes))
all_dur = list(set(dataset.del_t))

def load_taggers(_type='beatSim'): # Either beatSim or beat
    s1 = os.path.join(os.path.dirname(__file__), 'models' + '/' + _type + '/4gram_n_tagger_' +_type+'.pkl')
    s2 = os.path.join(os.path.dirname(__file__), 'models' + '/' + _type + '/4gram_t_tagger_'+_type+'.pkl')
    mn = joblib.load(s1)
    mt = joblib.load(s2)
    return (mn,mt)
    
def read_textfile(path):
    # Extracting data to predict and play
    text = []
    with open(path,'rb') as f:
        text = f.readlines()
        f.close()
    
    text = [t.decode()[:-2] for t in text[:-1]] + [text[-1].decode()]
    text = [t.split(' ') for t in text]
    
    pr_text = []
    for t in text:
        pr_text = pr_text + [t for t in t]
    
    print('----[INFO] Cleaning text data')
    pr_text = [dataset.preprocess(t) for t in pr_text]
    
    for i in range(0, (len(pr_text)-1)*2,2):
        pr_text.insert(i+1,'<l>') #inserting silent regions to add structure
    return pr_text

def predict_and_process(model_n,model_t,text):
    # Preprocessing input text
    print('----[INFO] Cleaning text data')
    text = [dataset.preprocess(t) for t in text]
    # Computing tag predictions
    notes = model_n.tag(text)
    dur = model_t.tag(text)
    
    #Postprocessing predictions to account for OOV words
    if notes[0][1] == None:
        # If we start None, not much we can do, so start C
        notes[0] = (notes[0][0], 'C2')
    
    if dur[0][1] == None:
        # If we start None, not much we can do, so start with 75ms
        dur[0] = (dur[0][0], '0')
        
    prev_n = notes[0] 
    prev_d = dur[0]
    i=0
    print('----[INFO] Filling in the gaps')
    for n,d in zip(notes,dur):
        if n[1] == None:
            # Choosing a ratio for now
            ratio = [1]
            f_prev = dataset.note2freq(prev_n[1])
            r = ratio[random.randrange(0,len(ratio))]
            if f_prev*r > 32767:
                f = f_prev * 1                
            else:
                f = f_prev * r
            notes[i] = (notes[i][0],dataset.pitch(f))
        if d[1] == None:
            dur[i] = (dur[i][0], '0')#(dur[i][0],all_dur[r])
            
        prev_n=notes[i]
        pred_d=dur[i]
        i=i+1
        
    return (notes,dur)
    # Preprocessing input text
    print('----[INFO] Cleaning text data')
    text = [dataset.preprocess(t) for t in text]
    # Computing tag predictions
    notes = model_n.tag(text)
    dur = model_t.tag(text)
    
    #Postprocessing predictions to account for OOV words
    if notes[0][1] == None:
        dur[0] = (dur[0][0],'0')#(dur[i][0],all_dur[r])
        # If we start None, not much we can do, so stat C
        notes[0] = (notes[0][0], 'C2')
    
    prev_n = notes[0] 
    i=0
    print('----[INFO] Filling in the gaps')
    for n,d in zip(notes,dur):
        if n[1] == None:
            dur[i] = (dur[i][0],'0')#(dur[i][0],all_dur[r])
            # Choosing a ratio for now
            ratio = [2,3,5]
            f_prev = dataset.note2freq(prev_n[1])
            r = ratio[random.randrange(0,len(ratio))]
            if f_prev*r > 32767:
                f = f_prev * 1                
            else:
                f = f_prev * r
            notes[i] = (notes[i][0],dataset.pitch(f))
    
        prev_n=notes[i]
        pred_d=dur[i]
        i=i+1
        
    return (notes,dur)

def play_text(text,_type='beatSim', tempo_scale=1, save=True):
    print('[INFO] Loading models')
    (mn,mt) = load_taggers(_type=_type)
    print('[INFO] Computing predictions')
    (notes,dur) = predict_and_process(mn,mt,text)
    print('[INFO] Playing music...')
    notes = [n[1] for n in notes]
    dur = [d[1] for d in dur]
    audio_stream = play_song.play(text,notes,dur,tutorial=False, tempo_scale=tempo_scale) # tutorial=True still not supported!
    return (notes,dur,audio_stream)

def __main__(path,_type,tempo_scale):
    # Reading textfile to perform prediction
    text = read_textfile(path)
    # Finalize conversion and play music!
    (notes,dur,audio_stream) = play_text(text,_type=_type,tempo_scale=tempo_scale)
    
    # Saving predictions to text file 
    out_file = ''
    for p in path:
        if p=='.':
            break
        out_file = out_file + p
        
    with open(out_file + '_converted.txt', 'w') as f:
        f.writelines([w + ':' + str((n,d)) + '\n' for w,n,d in zip(text,notes,dur)])
        f.close()
        
    Fs=44.1e3
    wavio.write(out_file+'.wav', audio_stream, Fs, sampwidth=3)

    
if __name__ == "__main__":
    path = input('Please enter the relative path to the .txt file you wish to convert: ')
    print('\n' + ' Please select the type of model you which to use for the conversion process')
    option = input("1: For 'The Beatles'" + "\n" + "2: 'The Beatles' and Similar: ")
    tempo_scale = input("Input an integer > 1 to scale the tempo by some fold: ")
    _type = ['beat', 'beatSim'][int(option)-1]
    print('[INFO] Starting conversion procedure')
    __main__(path,_type,eval(tempo_scale))
    print('[INFO] Done')

    