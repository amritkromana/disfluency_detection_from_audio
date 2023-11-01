import os
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed

def get_disf_desc(err_words, err_labels):
    # the split versions will have sublists defined by each phrase
    err_ip_idx = [i + 1 for i, val in enumerate(err_labels) if (val.endswith('IP')) or (val == 'FP')]
    if err_ip_idx[-1] < len(err_labels):
        err_ip_idx.append(len(err_labels))
    err_words_split = [err_words[i1:i2] for i1, i2 in zip([0] + err_ip_idx[:-1], err_ip_idx)]
    err_labels_split = [err_labels[i1:i2] for i1, i2 in zip([0] + err_ip_idx[:-1], err_ip_idx)]

    err_labels_split_nofp = [item for item in err_labels_split if item != ['FP']]
    if len(err_labels_split_nofp) > 1:
        ml = [1 if item != 'FP' else 0 for item in err_labels]
    else:
        ml = [0] * len(err_labels)

    ph = [[1] * len(sublist) if len(sublist) > 1 else [0] for sublist in err_labels_split]
    ph = [item for sublist in ph for item in sublist]

    ip = [1 if item.endswith('IP') else 0 for item in err_labels]

    pw = [1 if item.endswith('-') else 0 for item in err_words]

    return err_words_split, ml, ph, ip, pw


# convert IE label style to RP, RV, RS by comparing disf. to its correction
# also add F based on most commonly found filled pauses
def get_updated_labels(labels, words):
    updated_labels = pd.DataFrame(columns=['O', 'FP', 'C', 'RP', 'RV', 'RS', 'PW', 'PH', 'ML', 'IP'])

    for idx, lab in enumerate(labels):

        if len(updated_labels) > idx:
            continue

        # corrections, filled pauses, or outside of disfluencies
        if lab in ['C', 'FP', 'O']:
            updated_labels = updated_labels.append(
                pd.Series([1 if col == lab else 0 for col in updated_labels.columns], index=updated_labels.columns),
                ignore_index=True)

        # these labels suggest some type of disfluency
        # really, BE_IP and BE suggest the beginning of a disfluency, but sometimes they begin with these other labels
        elif lab in ['BE_IP', 'BE', 'IE', 'IP', 'C_IE', 'C_IP']:

            # try find the indices of the next repair and fluent word
            # if these don't exist, then set to None
            try:
                repair_idx = labels[idx:].index('C')
            except:
                repair_idx = None
            try:
                fluent_idx = labels[idx:].index('O')
            except:
                fluent_idx = None

            # restart: either there is no repair, or there is a fluent text
            # before a repair. this situation indicates that the next repair
            # is associated with a different reparanda.
            if repair_idx is None and fluent_idx is None:
                disf_type = 'RS'

                err_words = words[idx:]
                err_labels = labels[idx:]

                err_words_split, ml, ph, ip, pw = get_disf_desc(err_words, err_labels)

                o = [0 for item in err_labels]
                fp = [1 if item == 'FP' else 0 for item in err_labels]
                res = [1 if item != 'FP' else 0 for item in err_labels]

                disf_df = pd.DataFrame([o, o, fp, o, o, res, ph, ml, ip, pw],
                                       index=['O', 'C', 'FP', 'RP', 'RV', 'RS', 'PH', 'ML', 'IP', 'PW']).T
                updated_labels = updated_labels.append(disf_df, ignore_index=True)

            elif repair_idx is None or (fluent_idx is not None and fluent_idx < repair_idx):
                disf_type = 'RS'

                err_words = words[idx:idx + fluent_idx]
                err_labels = labels[idx:idx + fluent_idx]

                err_words_split, ml, ph, ip, pw = get_disf_desc(err_words, err_labels)

                o = [0 for item in err_labels]
                fp = [1 if item == 'FP' else 0 for item in err_labels]
                res = [1 if item != 'FP' else 0 for item in err_labels]

                disf_df = pd.DataFrame([o, o, fp, o, o, res, ph, ml, ip, pw],
                                       index=['O', 'C', 'FP', 'RP', 'RV', 'RS', 'PH', 'ML', 'IP', 'PW']).T
                updated_labels = updated_labels.append(disf_df, ignore_index=True)

            # repetition or revision other cases:
            else:
                err_words = words[idx:idx + repair_idx]
                err_labels = labels[idx:idx + repair_idx]

                err_words_split, ml, ph, ip, pw = get_disf_desc(err_words, err_labels)

                # corrections
                corr_words = words[idx + repair_idx:]
                corr_labels = labels[idx + repair_idx:]
                corr_end_idx = [i for i, val in enumerate(corr_labels) if val != 'C']
                if len(corr_end_idx) > 0:
                    corr_words = corr_words[:corr_end_idx[0]]

                # "you know" might be in the correction - we will not include these idxs in the rep/rev determinations
                corr_you_idx = [i for (i, item) in enumerate(corr_words) if item == "you"]
                corr_know_idx = [i for (i, item) in enumerate(corr_words) if item == "know"]
                corr_youknow_idx = [[i - 1, i] for i in corr_know_idx if i - 1 in corr_you_idx]
                corr_youknow_idx = [item for sublist in corr_youknow_idx for item in sublist]
                corr_oh_idx = [i for (i, item) in enumerate(corr_words[:1]) if item == "oh"]
                corr_well_idx = [i for (i, item) in enumerate(corr_words[:1]) if item == "well"]
                corr_yeah_idx = [i for (i, item) in enumerate(corr_words[:1]) if item == "yeah"]
                corr_like_idx = [i for (i, item) in enumerate(corr_words[:1]) if item == "like"]
                corr_skip_idx = corr_youknow_idx + corr_oh_idx + corr_well_idx + corr_yeah_idx + corr_like_idx
                corr_words = [item for i, item in enumerate(corr_words) if i not in corr_skip_idx]

                # determine rep or rev
                disf_type = 'RP'
                for err_sublist in err_words_split:
                    for err_word_idx, err_word in enumerate(err_sublist):

                        if err_word in ['uh', 'um']:
                            continue
                        if err_word_idx >= len(corr_words):
                            disf_type = 'RV'
                            break

                        if err_word.endswith('-') and corr_words[err_word_idx].startswith(err_word[:-1]):
                            continue
                        elif err_word == corr_words[err_word_idx]:
                            continue
                        else:
                            disf_type = 'RV'
                            break

                    if disf_type == 'RV':
                        break

                o = [0 for item in err_labels]
                fp = [1 if item == 'FP' else 0 for item in err_labels]
                rep = [1 if (disf_type == 'RP') and (item != 'FP') else 0 for item in err_labels]
                rev = [1 if (disf_type == 'RV') and (item != 'FP') else 0 for item in err_labels]

                disf_df = pd.DataFrame([o, o, fp, rep, rev, o, ph, ml, ip, pw],
                                       index=['O', 'C', 'FP', 'RP', 'RV', 'RS', 'PH', 'ML', 'IP', 'PW']).T
                updated_labels = updated_labels.append(disf_df, ignore_index=True)

    return updated_labels

def edit_labels():
    df = pd.read_csv('../data/transcripts.csv')

    df = df[['fname', 'speaker', 'seg_idx', 'word_idx', 'word', 'label', 'start', 'end']]

    start = time.time()

    df_grouped = df.groupby(['fname', 'speaker', 'seg_idx'])
    l_dfs = []
    skipped_turns = []

    for idx_turn, df_turn in df_grouped:

        # modify labels - ex fill in IP if it is missing
        df_turn['label_mod'] = df_turn['label']
        s_need_ip = (df_turn['label_mod'] == 'BE') & (~df_turn['label_mod'].shift(-1).isin(['IE', 'IP']))
        df_turn.loc[s_need_ip, 'label_mod'] = 'BE_IP'
        s_need_ip = (df_turn['label_mod'] == 'IE') & (~df_turn['label_mod'].shift(-1).isin(['IE', 'IP']))
        df_turn.loc[s_need_ip, 'label_mod'] = 'IP'
        s_need_ip = (df_turn['label_mod'] == 'C_IE') & (~df_turn['label_mod'].shift(-1).isin(['C_IE', 'C_IP']))
        df_turn.loc[s_need_ip, 'label_mod'] = 'C_IP'
        df_turn.loc[df_turn.word.isin(['uh', 'um']), 'label_mod'] = 'FP'

        # updated_labels will be populated based on words and labels in the loop
        labels = df_turn['label_mod'].tolist()
        words = df_turn['word'].tolist()

        try:
            updated_labels = get_updated_labels(labels, words)
            df_turn = df_turn.reset_index().merge(updated_labels, left_index=True, right_index=True)
            df_turn['E'] = df_turn[['RP', 'RV', 'RS']].sum(axis=1).astype(int)
            l_dfs.append(df_turn)
        except:
            print()
            skipped_turns.append(idx_turn)

    end = time.time()
    print(end - start)

    df_edited = pd.concat(l_dfs)
    df_edited.to_csv('../data/transcripts.csv')


def get_group_frame_labels(df_ngroup):

    start = df_ngroup.iloc[0]['start'] - 0.05
    df_ngroup['start'] = df_ngroup['start'] - start
    df_ngroup['end'] = df_ngroup['end'] - start
    end = df_ngroup.iloc[-1]['end'] + 0.05

    fname = df_ngroup['fname'].iloc[0].replace('.trans', '')
    speaker = df_ngroup['speaker'].iloc[0]
    seg_idx = df_ngroup['seg_idx'].iloc[0].astype(str).rjust(3, '0')

    label_fname = fname + '_' + speaker + seg_idx + '.npy'
    label_path = os.path.join('../data/labels_framelevel', label_fname)

    # we make S (silence) the first label followed by...
    labels = ['O', 'C', 'FP', 'RP', 'RV', 'RS', 'PW', 'E', 'IP']
    frame_time = np.arange(0, end, 0.01).tolist()
    # frame_gt = ['S'] * len(frame_time)
    frame_gt = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * len(frame_time)

    for idx, row in df_ngroup.iterrows():
        start_idx = round(row['start'] * 100)
        end_idx = round(row['end'] * 100)
        # frame_gt[start_idx:end_idx] = [row['updated_labels']] * (end_idx - start_idx)
        frame_gt[start_idx:end_idx] = [[0] + row[labels].values.tolist()] * (end_idx - start_idx)

    res = np.array(frame_gt)
    res = res[::2, ]

    np.save(label_path, res)

def get_frame_labels():

    df = pd.read_csv('../data/transcripts.csv')
    df['ngroup'] = df.groupby(['fname', 'speaker', 'seg_idx']).ngroup()
    ngroups = df['ngroup'].unique()

    label_path = os.path.join('../data/labels_framelevel')
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    params = [df.loc[df['ngroup'] == ngroup].copy() for ngroup in ngroups]
    Parallel(n_jobs=12)(delayed(get_group_frame_labels)(i) for i in params)
