from models import SourceFilterMixtureAutoencoder2
from utils import load_model

# Loading Model
tag = "unsupervised_2s_satb_bcbq_mf0_1"
umss_model = load_model(tag)
print("Model Params: ", umss_model)
