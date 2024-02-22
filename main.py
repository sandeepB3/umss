import librosa, librosa.display
from IPython.display import Audio
import torch
import torch.nn as nn
from models import Base_model, F0Extractor, SourceFilterMixtureAutoencoder2

audio1 = "./Datasets/ChoralSingingDataset/Nino_Dios/audio_16kHz/nino_Bajos_104.wav"
audio2 = "./Datasets/ChoralSingingDataset/Nino_Dios/audio_16kHz/nino_Soprano_105.wav"

signal1, sample_rate1 = librosa.load(audio1, sr=44100)
signal2, sample_rate2 = librosa.load(audio2, sr=44100)

# Play Audio 1
Audio(data=signal1, rate=sample_rate1)

# Play Audio 2
Audio(data=signal2, rate=sample_rate2)

# Network Dict Path for F0Extractor Model
network_dict_path = "./Datasets/ChoralSingingDataset/Nino_Dios/mixtures_2_sources/mf0_cuesta_processed/nino_dios_Sno_105_Bos_104.pt"

# Network Dict Path for SourceFilterMixtureAutoencoder2 Model
network_dict_path = "./trained_models/unsupervised_2s_satb_bcbq_mf0_1/unsupervised_2s_satb_bcbq_mf0_1.pth"

# Loading the SourceFilterMixtureAutoencoder2 Model
sourceFilterMixtureAutoEncoder2 = SourceFilterMixtureAutoencoder2()
state_dict = torch.load(network_dict_path)
sourceFilterMixtureAutoEncoder2.load_state_dict(state_dict)