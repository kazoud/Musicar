import numpy as np
import dataset
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import matplotlib.pyplot as plt

def generate_sin(freq, w_size, Fs):
    samples = np.linspace(0, w_size/Fs, w_size, endpoint=False)
    return np.sin(2*np.pi*freq*samples) * np.hamming(w_size)

def plot_signal(t, sig, xlabel='Time (sec)'):
    (fig, ax) = plt.subplots(1,1)
    fig.set_size_inches(18.5, 10.5)
    ax.plot(t, sig)
    ax.set_xlabel(xlabel)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    

def gen_audio_stream(notes, dur, Fs=44.1e3, tempo_scale=1):
    # Will generate an audio array to playback 
    freq_arr = [dataset.note2freq(n) for n in notes]
    dur_arr = tempo_scale * np.array([max(int(d)/1000,0) for d in dur])
    
    # Getting total number of samples expected
    total_dur = np.sum(dur_arr) # in seconds
    samples = int(np.ceil(total_dur * Fs))

    audio_stream = np.zeros(samples) 
    vis_stream = np.zeros(samples) 
    i = 0
    for f,d in zip(freq_arr, dur_arr):
        # Window size will be rounded off, whih can result in discrepancies so take 
        # this with a grain of salt
        w_size = int(round(d * Fs))
        sin = generate_sin(round(f), w_size, Fs)
        sin_vis = generate_sin(round(f/25), w_size, Fs)
        audio_stream[i:i+w_size] = sin[:len(audio_stream[i:i+w_size])]
        vis_stream[i:i+w_size] = sin_vis[:len(vis_stream[i:i+w_size])]
        i = i + w_size
    
    
    return (audio_stream, vis_stream)

    
def play(words, notes, dur, Fs=44.1e3, tutorial=False, tempo_scale=1):
    (audio_stream, vis_stream) = gen_audio_stream(notes, dur, Fs=Fs, tempo_scale=tempo_scale)
    if tutorial:
        animation.plot_song(vis_stream, words, notes, dur, Fs=Fs)
    else:
        (audio_stream, vis_stream) = gen_audio_stream(notes, dur, Fs=Fs, tempo_scale=tempo_scale)
        sd.play(audio_stream)
        for w,n,d in zip(words,notes,dur):
            print(w + ':' + n + ':' + d + ' x ' + str(tempo_scale))
            time.sleep(tempo_scale * int(d)/1000)        
        return audio_stream
    
def __main__():
    (words,notes,del_t,vocab,song_data) = dataset.get_data()
    song = song_data[-1]
    notes = [s[1] for s in song]
    dur = [s[2] for s in song]
    play(words, notes,dur, tutorial=False) # tutorial=True still not supported!

if __name__ == "__main__":
    __main__()