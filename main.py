from utils import load_model

# Loading Model
tag = "unsupervised_2s_CSD_mf0_1"
umss_model = load_model(tag)
print("Model Params: ", umss_model)
