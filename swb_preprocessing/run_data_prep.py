from align_files import align_files
from prep_audio import get_audio, add_silences
from prep_labels import edit_labels, get_frame_labels

if __name__ == '__main__':

    # get frame-level disfluency labels for analysis
    # check the README to ensure the ../raw_data directory is setup with
    # 1) audio, 2) transcripts, 3) outside/error/correction labels, and 4) the swapped speaker file
    # processed data will be printed in the ../data directory
    # a wav directory will have the audio
    # a wav_sil directory will have the audio augmented by 50 ms silence at the beginning and end
    # a transcripts.csv file will have all the word-level labels
    # a labels_framelevel directory will have all the frame-level labels

    # align disfluency + text files which contain slightly different tokenization schemes
    align_files()

    # get audio segment files based on segment timings
    get_audio()
    add_silences()

    # edit label disfluency classes and get them at the frame-level
    edit_labels()
    get_frame_labels()

