import torch

def tensor_from_tensor_list(tensor_list, excl_idx=None):
    toggle = False
    for i in range(len(tensor_list)):
        if excl_idx is not None:
            if i == excl_idx:
                continue
        if not toggle:
            tensor = tensor_list[i].detach().clone()
            toggle = True
            continue
        tensor = torch.concat((tensor, tensor_list[i].detach().clone()), dim=1)
    return tensor