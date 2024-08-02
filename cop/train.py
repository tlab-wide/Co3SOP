import argparse
import yaml
import torch
import importlib
from utils.data_builder import Ego, V2V

def train(model, dataloader, configs):
    pass

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', type=str, help='config path', default=
                           "configs/config.yaml")
    argparser.add_argument('--model', type=str, help='model name', default=
                           "SCPNet")
    args = argparser.parse_args()
    configs = yaml.load(args.config)
    model = importlib.import_module(f"models.{args.model}")
    if args.model != "COP-3D":
        dataloader = torch.utils.data.DataLoader(
            dataset=Ego,
            batch_size=configs["batch_size"],
            collate_fn=model.data_adaption,
            shuffle=configs["shuffle"],
            num_workers=configs["num_workers"]
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=V2V,
            batch_size=configs["batch_size"],
            collate_fn=model.data_adaption,
            shuffle=configs["shuffle"],
            num_workers=configs["num_workers"]
        )
    model.train(model.get_model(configs), dataloader, configs)