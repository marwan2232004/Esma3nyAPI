import torch


def get_device():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    return torch.device("cpu")


get_device()
