import librosa, librosa.display
from IPython.display import Audio
import torch
import torch.nn as nn
from utils import load_model

audio1 = "./Datasets/ChoralSingingDataset/Nino_Dios/audio_16kHz/nino_Bajos_104.wav"
audio2 = "./Datasets/ChoralSingingDataset/Nino_Dios/audio_16kHz/nino_Soprano_105.wav"

signal1, sample_rate1 = librosa.load(audio1, sr=44100)
signal2, sample_rate2 = librosa.load(audio2, sr=44100)

# Play Audio 1
Audio(data=signal1, rate=sample_rate1)

# Play Audio 2
Audio(data=signal2, rate=sample_rate2)

# Loading Model
tag = "unsupervised_2s_CSD_mf0_1"
umss_model = load_model(tag)
print("Model Params: ", umss_model)
