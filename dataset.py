import DALI as dali_code
import pandas as pd
from math import log2, pow
import numpy as np
import regex
import itertools

words = []
notes= []
del_t = []
vocab = []
song_data =[]

A4 = 440
C0 = A4*pow(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
octaves = [i for i in range(0,10)]
# all_notes = list(itertools.product(note_names, octaves))
# all_notes = [x[0]+str(x[1]) for x in all_notes]
base_freq = [-45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34]

word_lex = {'ye': 'yeah',
            'ya': 'yeah',
            'yea':'yeah',
            'yeah':'yeah',
            'yeh': 'yeah',
            'yah': 'yeah',
            'yeh': 'yeah',
            'oh': 'oh',
            'o': 'oh',
            'uh': 'uh',
            'ugh': 'uh'
            }

def pitch(freq):
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return note_names[n] + str(octave)

def note2freq(note):
    if note == 'N/A':
        return 1
    
    if '#' in note:
        i = note.index('#')
        n = note[:1]
        octave = int(note[2:])
    else:
        n = note[:1]
        octave = int(note[1:])
    
    base = 440
    
    idx = note_names.index(n)
    pos = base_freq[idx] + 12*(octave-1)
    freq = base*np.power(2, pos/12)
    return freq
    
def preprocess(word):
    # Making all words lower case
    clean = word.lower()
    # Removing random non-alphanumeric characters like hyphens or commas at the 
    # end of the word (preserve hyphens in the middle though)
    clean = regex.sub(r'\W*([\w\d-]*\w\b)\W*',r'\1',clean)
    # Dealing with word fillers like ooooo and uuuuugh 
    # ooooooh and ooooo
    patt1 = r'\b(o)+(h)*\b'
    sub1 = r'\1\2'
    # uuuuuuuuhhh and uuuuughhhh
    patt2 = r'\b(u)+(g)*(h)*\b'
    sub2 = r'\1\2\3'
    # yeaaahh and yeeeaa and yeeeeehhh
    patt3 = r'\b(y)+(e)*(a)*(h)*\b'
    sub3 = r'\1\2\3\4'
    # anything ending with yyyyy such as babyyyyy
    patt4 = r'([^y]+)(y)+'
    sub4 = r'\1\2'
    # NOTE: We don't want to normalize plurals since they are contextually important
    
    patterns = [patt1,patt2,patt3,patt4]
    subs = [sub1,sub2,sub3,sub4]
    
    for p,s in zip(patterns,subs):
        clean = regex.sub(p,s,clean)
        
    if clean in word_lex.keys():
        clean = word_lex[clean]
    
    # spell = SpellChecker()
    # clean = spell.correction(clean)
    
    return clean

def extract_sents(chosen_artists,dali_data, keys, dali_info):
    keys = list(dali_data.keys())
    #print(dali_info[0]) #-> array(['DALI_ID', 'NAME', 'YOUTUBE', 'WORKING'])
    
    # By artist
    print('----[INFO] Building dictionary of annotations for given artist')
    entry = {keys[i] : dali_data[keys[i]] for i in range(0, len(keys)) if dali_data[keys[i]].info['artist'] in chosen_artists}
    
    keys_b = list(entry.keys())
    
    dali_info_ev = [x for x in dali_info if x[0] in entry.keys()]
    song_names = [x[1] for x in dali_info_ev]
        
    # For downloading the audio 
    #path_audio = r'E:\AUB\EECE 634\Audio'
    #errors = dali_code.get_audio(dali_info_ev, path_audio, skip=[], keep=[])

    songs_data = []
    silence = '<l>'
    sents = []
    # Converting the frequency to notes
    k=0
    print('----[INFO] Processing tokens and annotations...')
    for (key,song_name) in zip(keys_b,song_names):
        song_ann = entry[key].annotations['annot']['words']
        #print(song_name)
        text = []
        notes = []
        time_delt = [] 
        index = []
        for i in range(0,len(song_ann)-1):
            ann = song_ann[i]
            ann_next = song_ann[i+1]
                    
            text = text + [preprocess(ann['text'])] + [silence]
            note_low = pitch(ann['freq'][0])
            note_high = pitch(ann['freq'][1])
            notes = notes + [(note_low,note_high)] + [('N/A','N/A')]
            time_delt = time_delt + [ann['time'][1]-ann['time'][0]] + [ann_next['time'][0]-ann['time'][1]]

            index = index + [ann['index']] + [ann['index']]
        
        text = text + [preprocess(ann_next['text'])]
        note_low = pitch(ann_next['freq'][0])
        note_high = pitch(ann_next['freq'][1])
        notes = notes + [(note_low,note_high)] + [('N/A','N/A')]
        time_delt = time_delt + [ann_next['time'][1]-ann_next['time'][0]]
        index = index + [ann_next['index']]
                     
        temp_t = []
    
        for t,b,td,i in zip(text,notes,time_delt,index):
            temp_t = temp_t + [(t,b[-1],str(int(td*1000)),i)]
 
        temp = {}
        for i in index:
            s = [s for s in temp_t if s[-1]==i]
            temp.update({i:s})
        
        sents.append(temp)
        songs_data = songs_data + [temp_t]
        k=k+1
    
    print('----[INFO] Done')        
    return (songs_data,sents)

def get_dali_data():
    dali_data_path = r'D:\AUB\EECE 634\DALI'
    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])
    dali_info = dali_code.get_info(dali_data_path + '\info\DALI_DATA_INFO.gz')
    return (dali_data,dali_info)

def __main__():
    print('[INFO] Extracting DALI dataset')
    (dali_data,dali_info) = get_dali_data()
    keys = list(dali_data.keys())
    artists = [dali_data[keys[i]].info['artist'] for i in range(0,len(keys))]
    # Extracting similar artists to the beatles from predefined text file
    sim = []
    with open('similar.txt','rb') as f:
        sim = f.readlines()
        f.close()
    similar_artists = [s.decode()[:-2] for s in sim[:-1]] + [sim[-1].decode()]
    
    chosen_artists = ['The Beatles'] + [s for s in similar_artists if s in artists]
    
    print('[INFO] Reading tokens and annotations...')
    (song_data, sents) = extract_sents(chosen_artists,dali_data,keys,dali_info)
    # Getting total flattened set of sentences
    sentences = []
    for sent in sents:
        sentences = sentences + list(sent.values())
    
    print('[INFO] Computing unigrams and vocabulary')
    unigrams = []
    for s in song_data:
        unigrams = unigrams + [x for x in s]
        
    words = list([preprocess(x[0]) for x in unigrams])
    notes = list([x[1] for x in unigrams])
    del_t = list([x[2] for x in unigrams])
    vocab = set(words)
    
    print('[INFO] Vocab Size: ' + str(len(vocab)))
    print('[INFO] Corpus Size: ' + str(len(unigrams)))
    return (words,notes,del_t,vocab,song_data)

if __name__ == "__main__":
    print('[INFO] Starting dataset extraction')
    (words,notes,del_t,vocab,song_data) = __main__()
    print('[INFO] Done')
    
def get_data():
    print('[INFO] Retrieving data')
    return __main__()
    
