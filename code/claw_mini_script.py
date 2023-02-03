import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import Model_Cond_Diffusion, Model_cnn_mlp
from train import ClawCustomDataset

DATASET_PATH = "dataset"
SAVE_DATA_DIR = "output"  # for models/data
SCATTER_KWARGS = {"s": 1, "alpha": 0.25, "color": "yellow"}
SAVE_FIGURE_DIR = "figures"

os.makedirs(SAVE_FIGURE_DIR, exist_ok=True)

n_epoch = 40
lrate = 1e-3
device = "cuda"
n_hidden = 128
batch_size = 32
n_T = 50
net_type = "fc"
drop_prob = 0.0
extra_diffusion_steps = 16

# get datasets
tf = transforms.Compose([])
torch_data_train = ClawCustomDataset(
    DATASET_PATH, transform=tf, train_or_test="train", train_prop=0.90
)
dataload_train = DataLoader(
    torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
)

x_shape = torch_data_train.image_all.shape[1:]
y_dim = torch_data_train.action_all.shape[1]

# set up model
nn_model = Model_cnn_mlp(
    x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
).to(device)
model = Model_Cond_Diffusion(
    nn_model,
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    x_dim=x_shape,
    y_dim=y_dim,
    drop_prob=drop_prob,
    guide_w=0.0,
)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lrate)

# main training loop
for ep in tqdm(range(n_epoch), desc="Epoch"):
    results_ep = [ep]
    model.train()

    # lrate decay
    optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

    # train loop
    pbar = tqdm(dataload_train)
    loss_ep, n_batch = 0, 0
    for x_batch, y_batch in pbar:
        x_batch = x_batch.type(torch.FloatTensor).to(device)
        y_batch = y_batch.type(torch.FloatTensor).to(device)
        loss = model.loss_on_batch(x_batch, y_batch)
        optim.zero_grad()
        loss.backward()
        loss_ep += loss.detach().item()
        n_batch += 1
        pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
        optim.step()
    results_ep.append(loss_ep / n_batch)


# eval, plot samples of certain inputs
model.eval()
idxs = [14, 2, 0, 9, 5, 35, 16] # interesting data egs
idxs_data = [[] for _ in range(len(idxs))]
fig, axs = plt.subplots(nrows=3, ncols=len(idxs))
for i, idx in enumerate(idxs):
    x_eval = (
        torch.Tensor(torch_data_train.image_all[idx])
        .type(torch.FloatTensor)
        .to(device)
    )
    x_eval_large = torch_data_train.image_all_large[idx]
    obj_mask_eval = torch_data_train.label_all[idx]

    for j in range(6): # can't fit batch of 300 into memory, so do it 6x50
        x_eval_ = x_eval.repeat(50, 1, 1, 1)
        with torch.set_grad_enabled(False):

            if extra_diffusion_steps == 0:
                y_pred_ = (
                    model.sample(x_eval_, extract_embedding=True).detach().cpu().numpy()
                )
            else:
                y_pred_ = model.sample_extra(x_eval_, extra_steps=extra_diffusion_steps).detach().cpu().numpy()
        if j == 0:
            y_pred = y_pred_
        else:
            y_pred = np.concatenate([y_pred, y_pred_])
    x_eval = x_eval.detach().cpu().numpy()
    axs[0,i].imshow(x_eval_large)
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])
    axs[0,i].set_ylim(63, -1)
    axs[0,i].set_xlim(63, -1)

    # ground truth
    axs[1,i].imshow(obj_mask_eval, vmin=0, vmax=1)
    axs[1,i].set_xticks([])
    axs[1,i].set_yticks([])
    axs[1,i].set_ylim(63, -1)
    axs[1,i].set_xlim(63, -1)

    # feint ground truth and prediction
    axs[2,i].imshow(obj_mask_eval*0.3, vmin=0, vmax=1)
    axs[2,i].scatter(
        y_pred[:, 0] * 64,
        y_pred[:, 1] * 64,
        **SCATTER_KWARGS
    )
    axs[2,i].set_xticks([])
    axs[2,i].set_yticks([])
    axs[2,i].set_ylim(63, -1)
    axs[2,i].set_xlim(63, -1)
fig.savefig(os.path.join(SAVE_FIGURE_DIR, "claw_mini_diffusion_eg.png"))


