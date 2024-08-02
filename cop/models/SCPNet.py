import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)))+"/includes/Codes-for-SCPNet")
from builder import model_builder


model = None

def get_model(configs):
    global model
    if model is None:
        model = model_builder.build(configs["model_params"]["SCPNet"])
    return model

def data_adaption()

