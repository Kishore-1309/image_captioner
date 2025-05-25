import os
import torch

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Handle DataParallel or DDP wrapped model
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    torch.save(checkpoint, path)
    return path

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    # Load weights correctly whether or not model is wrapped with DataParallel
    if hasattr(model, 'module'):
        # Model is wrapped with DataParallel
        for k, v in state_dict.items():
            new_state_dict[f"module.{k}" if not k.startswith("module.") else k] = v
    else:
        # Model is not wrapped, remove "module." prefix if present
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(new_state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch']
