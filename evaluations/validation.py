import torch
from evaluations import DiceLoss

def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    criterion = DiceLoss()
    for batch in loader:
        imgs, labels = batch['image'], batch['label']
        imgs = imgs.to(device=device, dtype=torch.float32)

        labels = labels.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = net(imgs)
            tot += criterion(pred, labels).item()

    net.train()
    return tot / n_val
