import torch
import torchaudio
from torchaudio.transforms import Resample
from models import SourceFilterMixtureAutoencoder2
from utils import load_model

# Loading Model
tag = "unsupervised_2s_satb_bcbq_mf0_1"
umss_model = load_model(tag)
print("Model Params: ", umss_model, "\n")

# Loading mixed audio file
audio_file = "./Datasets/ChoralSingingDataset/Nino_Dios/mixtures_2_sources/mixed_audio/nino_bajos104_sop_105.wav"
mixed_audio, sample_rate = torchaudio.load(audio_file)

resample = Resample(orig_freq=sample_rate, new_freq=16000) # architecture is designed for signals sampled at 16kHz
mixed_audio_resampled = resample(mixed_audio)

input_audio = mixed_audio_resampled.squeeze(1) # match the expected shape [batch_size, n_samples]

print("Input Audio:\n")
print("Input Audio Dimesnion: ", input_audio.size(),"\n")
print("Input Audio Info: ", input_audio,"\n")

# Loading the F0 information of nino_dios_Sno_105_Bos_104.pt file
f0_file = "./Datasets/ChoralSingingDataset/Nino_Dios/mixtures_2_sources/mf0_cuesta_processed/nino_dios_Sno_105_Bos_104.pt"
input_f0 = torch.load(f0_file)
input_f0 = input_f0.unsqueeze(0) #Ensuring the shape is appropriate for your model ([batch_size, n_freq_frames, n_sources])

print("Input F0:\n")
print("Input F0 Dimesnion: ", input_f0.size(),"\n")
print("Input F0 Info: ", input_f0,"\n")

# Perform inference
with torch.no_grad():
    output = umss_model(input_audio, input_f0)

# Process output
if isinstance(output, tuple):
    mix, sources = output
    # Process separated sources as needed
    print("Separated sources shape:", sources.shape)
else:
    mix = output
    print("Mixed audio shape:", mix.shape)