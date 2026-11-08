import os
import numpy as np
import torch


def prRed(prt):
    print("\033[91m {}\033[00m".format(prt))


def prGreen(prt):
    print("\033[92m {}\033[00m".format(prt))


def prYellow(prt):
    print("\033[93m {}\033[00m".format(prt))


def to_numpy(x):
    return x.cpu().detach().double().numpy()


def to_tensor(x, dtype, device, requires_grad=False):
    if type(x).__module__ != "numpy":
        return x
    return torch.from_numpy(x).type(dtype).to(device).requires_grad_(requires_grad)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir, env_name):
    """Create and return the next run directory under parent_dir."""
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        try:
            folder_id = int(folder_name.split("-run")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except Exception:
            pass

    experiment_id += 1
    run_dir = os.path.join(parent_dir, "{}-run{}".format(env_name, experiment_id))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
