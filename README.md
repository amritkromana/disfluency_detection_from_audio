# Disfluency Detection From Audio 

This repo includes a demo for running audio through the language, acoustic, and multimodal disfluency detection models. 
It also includes preprocessing code for the Switchboard dataset, to align audio, transcripts, and disfluency labels 
(filled pauses, partial words, repetitions, revisions, and restarts) at the frame-level (every 20ms of audio). 

# Disfluency Detection Demo

## Dependencies 

The following packages are needed: 
- pandas 
- torch 
- transformers 
- whisper_timestamped 
- gdown 

Use gdown to download the pretrained model weights and save to demo_models: 
```
mkdir demo_models && cd demo_models
mkdir asr && cd asr
gdown --id 1BeT7m_5qv19Sb5yrZ2zhKu6fEprUoB9N -O config.json
gdown --id 15xQiVew2SatAL_7E5Hh30hya8x7tyGb_ -O pytorch_model.bin
cd ..
gdown --id 1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX -O language.pt
gdown --id 1wWrmopvvdhlBw-cL7EDyih9zn_IJu5Wr -O acoustic.pt
gdown --id 1LPchbScA_cuFx1XoNxpFCYZfGoJCfWao -O multimodal.pt
```

## How to run the demo 

Given some input.wav and output.csv, we can run any of these options: 
```
python3 demo.py --input_file input.wav --output_file output.csv --modality language
```
```
python3 demo.py --input_file input.wav --output_file output.csv --modality acoustic
```
```
python3 demo.py --input_file input.wav --output_file output.csv --modality multimodal
```
The ``language`` option runs a Whisper model that's been fine-tuned for verbatim transcription, and then uses the text + timestamps as input to a BERT model that's been fine-tuned for disfluency detection.
The ``acoustic`` option runs a WavLM model that's been fine-tuned for acoustic-based disfluency detection. 
The ``multimodal`` option runs the language and acoustic models, concatenates their embeddings, and runs them through a BLSTM fusion model. 
The frame-level disfluency predictions will be printed to output.csv. 

# Switchboard Preprocessing 

## Dependencies 

The following packages are needed:
- pandas 
- Levenshtein

Prepare the data as follows: 
- Get switchboard data through LDC: 
  - Copy audio sph files to raw_data folder (raw_data/swb_sph)
  - Copy ms98 transcriptions to raw_data folder (raw_data/swb_ms98_transcriptions)
- Copy corrected disfluency labels from Zayats et al. to raw_data folder: 
```
cd raw_data
wget https://raw.githubusercontent.com/vickyzayats/switchboard_corrected_reannotated/master/switchboard_corrected_with_silver_reannotation.zip
unzip switchboard_corrected_with_silver_reannotation.zip
mv switchboard_corrected_with_silver_reannotation.tsv swb_silver.tsv
rm switchboard_corrected_with_silver_reannotation.zip
```

## How to run the preprocessing code
```
python3 run_data_prep.py 
```
This will create a data folder with 
- transcripts.csv: the text and word-level disfluency labels associated with each segment (FP, RP, RV, RS, PW)
- wav_sil: a directory with the 8k wav files associated with each segment (50 ms silence padding on either end)
- labels_framelevel: a directory with the frame-level labels (labels for every 20 ms) 

# Citation 
This work has been submitted to IEEE Transactions on Audio, Speech and Language Processing. If you use this work in your research or projects, please cite it as follows:
```
@article{romana2023,
title = {Automatic Disfluency Detection from Untranscribed Speech},
author = {Amrit Romana, Kazuhito Koishida, Emily Mower Provost},
year = {2023}
}
```
