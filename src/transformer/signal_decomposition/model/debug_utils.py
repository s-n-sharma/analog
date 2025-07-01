import matplotlib.pyplot as plt
import torch

def plot_signals(mixed, pred, true, idx=0):
    """
    Plot mixed signal, predicted sources, and true sources for a given batch index.
    mixed: (batch, seq_len)
    pred: (batch, 2, seq_len)
    true: (batch, 2, seq_len)
    """
    if isinstance(mixed, torch.Tensor):
        mixed = mixed.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(mixed[idx], label='Mixed')
    plt.title('Mixed Signal')
    plt.subplot(3, 1, 2)
    plt.plot(true[idx, 0], label='True Source 1')
    plt.plot(true[idx, 1], label='True Source 2')
    plt.title('True Sources')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(pred[idx, 0], label='Pred Source 1')
    plt.plot(pred[idx, 1], label='Pred Source 2')
    plt.title('Predicted Sources')
    plt.legend()
    plt.tight_layout()
    plt.show() 