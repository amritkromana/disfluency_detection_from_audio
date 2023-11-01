import os, ast, re
import pandas as pd
import Levenshtein as lev

def scrub_word(word):

    # text preprocessing needed to align the disfluency labels text with the word timings text

    word = word.replace('<b_aside>', '').replace('<e_aside>', '')

    if re.match("(\[\w+-)?([0-9a-zA-Z-']+)(\[[0-9a-zA-Z-']+\])(-?)(\])", word) is not None:
        word = re.sub("(\[\w+-)?([0-9a-zA-Z-']+)(\[[0-9a-zA-Z-']+\])(-?)(\])", "\\2" + "\\4", word)
    elif re.match("[\[]\w+-[0-9a-zA-Z-']+[\]]", word) is not None:
        word = re.sub("[\[]\w+-([0-9a-zA-Z-']+)[\]]", "\\1", word)
    else:
        word = re.sub("[\[].*?[\]]", "", word)

    word = re.sub("(\w+)(_.*)", "\\1", word)  # _1 -->

    word = re.sub("\{([0-9a-zA-Z-']+)\}", "\\1", word)  # {federale} --> federale

    word = re.sub("-\/([0-9a-zA-Z-']+)\]", "\\1", word)  # [csh[ape]-/shape] --> shape
    word = re.sub("(\w+)/(\w+)\]", "\\2", word)  # [-[Can]sego/Canseco] --> Canseco

    word = word.lower()

    return word

def align_files():

    # read in disfluency file
    df = pd.read_csv('../raw_data/swb_silver.tsv', delimiter='\t')
    fnames = df['file'].unique()

    # there are some speaker inconsistencies between the disfluency + timings files
    # swb_swapped_speak.txt contains a list of files where speaker A + B need to be reversed
    with open('../raw_data/swb_swapped_speak.txt', 'r') as fp:
        swapped_speakers = fp.readlines()
    swapped_speakers = [speaker.strip('\n') for speaker in swapped_speakers]

    # loop through each audio file, speaker, utterance to get the audio, text, + labels
    df_results = pd.DataFrame()
    seg_skipped = []
    for audio_fname in fnames:
        df_audio = df.loc[df['file'] == audio_fname]

        if audio_fname in swapped_speakers:
            df_audio['speaker'] = df_audio['speaker'].map({'A': 'B', 'B': 'A'})

        for speaker in ['A', 'B']:
            df_sub = df_audio.loc[df_audio['speaker'] == speaker]

            seg_bound = []
            seg_start = 0
            words_disf = []
            labels = []

            # preprocess the disfluency data: make tokenization more consistent with timings file
            for utt_idx, row in df_sub.iterrows():

                tmp_words = ast.literal_eval(row['ms_sentence'])
                tmp_labels = ast.literal_eval(row['ms_disfl'])

                tmp_words = [word for word in tmp_words if word != '--' and word != '//']

                rem_wordidx = []
                for word_idx in range(len(tmp_words) - 1):
                    if tmp_words[word_idx + 1].startswith("'") or tmp_words[word_idx + 1] == "n't":
                        tmp_words[word_idx] = tmp_words[word_idx] + tmp_words[word_idx + 1]
                        rem_wordidx.append(word_idx + 1)
                tmp_words = [word for idx, word in enumerate(tmp_words) if idx not in rem_wordidx]
                tmp_labels = [label for idx, label in enumerate(tmp_labels) if idx not in rem_wordidx]

                if len(tmp_words) > 0:
                    if len(tmp_words) != len(tmp_labels):
                        print('len(tmp_words) != len(tmp_labels)')

                    words_disf.extend(tmp_words)
                    labels.extend(tmp_labels)
                    seg_bound.append((seg_start, seg_start + len(tmp_words)))
                    seg_start += len(tmp_words)

            seg_lens = [val[1] - val[0] for val in seg_bound]

            # preprocess the timings data: "scrub" words
            trans_fname = os.path.join('../raw_data/swb_ms98_transcriptions',
                                       audio_fname[2:4], audio_fname[2:6],
                                       audio_fname[:6] + speaker + '-ms98-a-word.text')
            with open(trans_fname, 'r') as fp:
                trans = fp.readlines()

            trans = [re.sub('\s+', ' ', trans_line.strip('\n')) for trans_line in trans]
            words_tim = [scrub_word(trans_line.split(' ')[-1]) for trans_line in trans]
            timings = [(trans_line.split(' ')[1], trans_line.split(' ')[2]) for trans_line in trans]
            rem_idx = [idx for idx, val in enumerate(words_tim) if val == '']
            words_tim = [val for idx, val in enumerate(words_tim) if idx not in rem_idx]
            timings = [val for idx, val in enumerate(timings) if idx not in rem_idx]

            # merge/align the disfluency + timing data
            # if a word does not exist in the word_tim list that means we don't have timing info
            # skip this entire segment
            r = re.compile(r"[\w'\-\&\/]+")
            words_disf_dic = {i: (m.start(0), m.group(0)) for i, m in enumerate(r.finditer(' '.join(words_disf)))}
            words_tim_dic = {i: (m.start(0), m.group(0)) for i, m in enumerate(r.finditer(' '.join(words_tim)))}
            editops = lev.editops(' '.join(words_disf), ' '.join(words_tim))

            while len(editops) > 0:

                editop, sidx, didx = editops[0]

                if editop in ['insert', 'replace']:
                    ins_word_idx = [key for key, val in words_tim_dic.items() if val[0] <= didx][-1]
                    words_tim.pop(ins_word_idx)
                    timings.pop(ins_word_idx)

                if editop in ['delete', 'replace']:

                    del_word_idx = [key for key, val in words_disf_dic.items() if val[0] <= sidx][-1]
                    words_disf.pop(del_word_idx)
                    labels.pop(del_word_idx)
                    seg_bound_update = []
                    for seg_start, seg_end in seg_bound:
                        if seg_start >= del_word_idx:
                            seg_start -= 1
                        if seg_end >= del_word_idx:
                            seg_end -= 1
                        seg_bound_update.append((seg_start, seg_end))
                    seg_bound = seg_bound_update.copy()

                words_disf_dic = {i: (m.start(0), m.group(0)) for i, m in enumerate(r.finditer(' '.join(words_disf)))}
                words_tim_dic = {i: (m.start(0), m.group(0)) for i, m in enumerate(r.finditer(' '.join(words_tim)))}
                editops = lev.editops(' '.join(words_disf), ' '.join(words_tim))

            if len(words_tim) != len(words_disf):
                print('final lengths dont match')

            for seg_idx, (seg_start, seg_end) in enumerate(seg_bound):

                if (seg_end - seg_start) != seg_lens[seg_idx]:
                    seg_skipped.append((audio_fname, speaker, seg_idx))
                    continue

                seg_fnames = [audio_fname] * (seg_end - seg_start)
                seg_speakers = [speaker] * (seg_end - seg_start)
                seg_seg_idxs = [seg_idx] * (seg_end - seg_start)
                seg_word_idxs = list(range(seg_end - seg_start))
                seg_words = words_disf[seg_start:seg_end]
                seg_labels = labels[seg_start:seg_end]
                seg_starts = [seg_timings[0] for seg_timings in timings[seg_start:seg_end]]
                seg_ends = [seg_timings[1] for seg_timings in timings[seg_start:seg_end]]

                df_tmp = pd.DataFrame(
                    [seg_fnames, seg_speakers, seg_seg_idxs, seg_word_idxs, seg_words, seg_labels, seg_starts, seg_ends],
                    index=['fname', 'speaker', 'seg_idx', 'word_idx', 'word', 'label', 'start', 'end']).T

                df_results = pd.concat([df_results, df_tmp])

    output_dir = os.path.join('..', 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_results.to_csv('../data/transcripts.csv')

    df_results['start'] = df_results['start'].astype(float)
    df_results['end'] = df_results['end'].astype(float)

    # check that the durations are all nonnegative
    df_grped = df_results.groupby(['fname', 'speaker', 'seg_idx'])
    if min(df_grped['end'].max() - df_grped['start'].min()) < 0:
        print('error: negative duration segment')

    # check that start time is always increasing
    if len(df_grped) != df_grped['start'].is_monotonic_increasing.sum():
        print('error: start time is decreasing within a segment')

    # check that end time is always increasing
    if len(df_grped) != df_grped['end'].is_monotonic_increasing.sum():
        print('error: end time is decreasing within a segment')
    if min((df_results['start'] - df_grped['end'].shift()).dropna()) < 0:
        print('error: words overlap')

