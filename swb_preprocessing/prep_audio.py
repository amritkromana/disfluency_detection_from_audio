import os
import pandas as pd
from joblib import Parallel, delayed

def get_one_audio(idx, row):
    fname, speaker, seg_idx = idx

    sph_fname = fname[:2] + '0' + fname[2:].replace('.trans', '.sph')
    input_fname = os.path.join('..', 'raw_data', 'swb_sph', sph_fname)

    wav_fname = fname.replace('.trans', '') + '_' + speaker + str(seg_idx).rjust(3, '0') + '.wav'
    output_fname = os.path.join('..', 'data', 'wav', wav_fname)

    if os.path.exists(output_fname):
        return

    if speaker == 'A':
        channel = '1'
    else:
        channel = '2'

    start = str(row['start'])
    dur = str(row['end'] - row['start'])

    cmd = 'sox ' + input_fname + ' ' + output_fname + ' remix ' + channel + ' trim ' + start + ' ' + dur
    os.system(cmd)

def get_audio():
    df = pd.read_csv('../data/transcripts.csv')
    output_dir = os.path.join('..', 'data', 'wav')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_grped = df.groupby(['fname', 'speaker', 'seg_idx'])
    df_seg = df_grped.agg({'start': 'min', 'end': 'max', 'word': ' '.join})

    params = [[idx, df_seg.loc[idx]] for idx in df_seg.index.values]
    Parallel(n_jobs=8)(delayed(get_one_audio)(a, b) for a, b in params)

def add_one_silence(fname):
    input_dir = os.path.join('..', 'data', 'wav')
    output_dir = os.path.join('..', 'data', 'wav_sil')
    sil_path = os.path.join('..', 'data', 'silence.wav')

    input_fname = os.path.join(input_dir, fname)
    output_fname = os.path.join(output_dir, fname)

    if os.path.exists(output_fname):
        return

    cmd = 'sox ' + sil_path + ' ' + input_fname + ' ' + sil_path + ' ' + output_fname
    os.system(cmd)

def add_silences():

    input_dir = os.path.join('..', 'data', 'wav')
    sil_path = os.path.join('..', 'data', 'silence.wav')
    output_dir = os.path.join('..', 'data', 'wav_sil')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = 'sox -n -r 8000 -c 1 ' + sil_path + ' trim 0.0 0.05'
    os.system(cmd)

    params = [f for f in os.listdir(input_dir)]
    Parallel(n_jobs=8)(delayed(add_one_silence)(a) for a in params)

