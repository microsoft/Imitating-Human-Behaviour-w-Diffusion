from argparse import ArgumentParser
import os
from collections import OrderedDict
from itertools import product

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from models import (
    Model_cnn_mlp,
    Model_Cond_Discrete,
    Model_Cond_MSE,
    Model_Cond_MeanVariance,
    Model_Cond_Diffusion,
    Model_Cond_BeT,
    Model_Cond_Kmeans,
    Model_Cond_EBM
)

DATASET_PATH = "dataset"
SAVE_DATA_DIR = "output"  # for models/data

EXPERIMENTS = [
    {
        "exp_name": "diffusion",
        "model_type": "diffusion",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "mse",
        "model_type": "mse",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "discrete",
        "model_type": "discrete",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "meanvariance",
        "model_type": "meanvariance",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "kmeans",
        "model_type": "kmeans",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "bet",
        "model_type": "bet",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "cfg",
        "model_type": "diffusion",
        "drop_prob": 0.1,
    },
    {
        "exp_name": "ebm_derivative_free",
        "model_type": "ebm",
        "sample_mode": "derivative_free",
        "drop_prob": 0.0,
    },
    {
        "exp_name": "ebm_langevin",
        "model_type": "ebm",
        "sample_mode": "langevin",
        "drop_prob": 0.0,
    },
]

EXTRA_DIFFUSION_STEPS = [0, 2, 4, 8, 16, 32]
GUIDE_WEIGHTS = [0.0, 4.0, 8.0]

n_epoch = 100
lrate = 1e-4
device = "cuda"
n_hidden = 512
batch_size = 32
n_T = 50
net_type = "transformer"


class ClawCustomDataset(Dataset):
    def __init__(
        self, DATASET_PATH, transform=None, train_or_test="train", train_prop=0.90
    ):
        self.DATASET_PATH = DATASET_PATH
        # just load it all into RAM
        self.image_all = np.load(os.path.join(DATASET_PATH, "images_small.npy"), allow_pickle=True)
        self.image_all_large = np.load(os.path.join(DATASET_PATH, "images.npy"), allow_pickle=True)
        self.label_all = np.load(os.path.join(DATASET_PATH, "labels.npy"), allow_pickle=True)
        self.action_all = np.load(os.path.join(DATASET_PATH, "actions.npy"), allow_pickle=True)
        self.transform = transform
        n_train = int(self.image_all.shape[0] * train_prop)
        if train_or_test == "train":
            self.image_all = self.image_all[:n_train]
            self.label_all = self.label_all[:n_train]
            self.action_all = self.action_all[:n_train]
        elif train_or_test == "test":
            self.image_all = self.image_all[n_train:]
            self.label_all = self.label_all[n_train:]
            self.action_all = self.action_all[n_train:]
        else:
            raise NotImplementedError

        # normalise actions and images to range [0,1]
        self.action_all = self.action_all / 64.0
        self.image_all = self.image_all / 255.0

    def __len__(self):
        return self.image_all.shape[0]

    def __getitem__(self, index):
        image = self.image_all[index]
        action = self.action_all[index]
        if self.transform:
            image = self.transform(image)
        return (image, action)


def main(experiment):
    # Unpack experiment settings
    exp_name = experiment["exp_name"]
    model_type = experiment["model_type"]
    drop_prob = experiment["drop_prob"]

    # get datasets set up
    tf = transforms.Compose([])
    torch_data_train = ClawCustomDataset(
        DATASET_PATH, transform=tf, train_or_test="train", train_prop=0.90
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
    )

    x_shape = torch_data_train.image_all.shape[1:]
    y_dim = torch_data_train.action_all.shape[1]

    # EBM langevin requires gradient for sampling
    requires_grad_for_eval = False

    # create model
    if model_type == "diffusion":
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
    elif model_type == "mse":
        nn_model = Model_cnn_mlp(
            x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
        ).to(device)
        model = Model_Cond_MSE(nn_model, device=device, x_dim=x_shape, y_dim=y_dim)
    elif model_type == "discrete":
        n_bins = 20  # how many bins to discretise each action dimension into, assumes actions in [-1, 1]
        nn_model = Model_cnn_mlp(
            x_shape,
            n_hidden,
            y_dim,
            embed_dim=128,
            output_dim=n_bins * y_dim,
            net_type=net_type,
        ).to(device)
        model = Model_Cond_Discrete(
            nn_model, device=device, x_dim=x_shape, y_dim=y_dim, n_bins=n_bins
        )
    elif model_type == "meanvariance":
        # output mean and variance of a Gaussian
        nn_model = Model_cnn_mlp(
            x_shape, n_hidden, y_dim, embed_dim=128, output_dim=2 * y_dim, net_type=net_type
        ).to(device)
        model = Model_Cond_MeanVariance(nn_model, device=device, x_dim=x_shape, y_dim=y_dim)
    elif model_type == "kmeans":
        # first cluster as kmeans over whole dataset, then treat as classification problem
        n_k = 10
        kmeans_model = KMeans(n_clusters=n_k, random_state=2).fit(
            torch_data_train.action_all
        )
        nn_model = Model_cnn_mlp(
            x_shape, n_hidden, y_dim, embed_dim=128, output_dim=n_k, net_type=net_type
        ).to(device)
        model = Model_Cond_Kmeans(
            nn_model, device=device, x_dim=x_shape, y_dim=y_dim, kmeans_model=kmeans_model
        )
    elif model_type == "bet":
        # first cluster as kmeans over whole dataset, then treat as classification problem
        # we need n_k*(y_dim+1) outputs, since we need n_k classification tasks, then n_k*y_dim options for the residual
        n_k = 10
        kmeans_model = KMeans(n_clusters=n_k, random_state=2).fit(
            torch_data_train.action_all
        )
        nn_model = Model_cnn_mlp(
            x_shape,
            n_hidden,
            y_dim,
            embed_dim=128,
            output_dim=n_k * (y_dim + 1),
            net_type=net_type,
        ).to(device)
        model = Model_Cond_BeT(
            nn_model, device=device, x_dim=x_shape, y_dim=y_dim, kmeans_model=kmeans_model
        )
    elif model_type == "ebm":
        sample_mode = experiment["sample_mode"]
        nn_model = Model_cnn_mlp(
            x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type, output_dim=1
        ).to(device)
        if sample_mode == "derivative_free":
            model = Model_Cond_EBM(
                nn_model,
                device=device,
                x_dim=x_shape,
                y_dim=y_dim,
                n_counter_egs=256,
                ymin=0,
                ymax=1,
                n_samples=128,
                n_iters=3,
                stddev=0.33,
                K=0.5,
                sample_mode=sample_mode,
            )
        elif sample_mode == "langevin":
            model = Model_Cond_EBM(
                nn_model,
                device=device,
                x_dim=x_shape,
                y_dim=y_dim,
                n_counter_egs=32,
                ymin=0,
                ymax=1,
                n_samples=128,
                n_iters=3,
                stddev=0.33,
                K=0.5,
                sample_mode=sample_mode
            )
            model.l_noise_scale = 0.5
            model.l_coeff_start = 0.05
            model.l_coeff_end = 0.005
            model.n_mcmc_iters = 50
            model.l_n_sampleloops = 2
            # whether to overwrite half of langevin samples with uniform
            model.l_overwrite = False
            requires_grad_for_eval = True
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)

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

    model.eval()
    idxs = [14, 2, 0, 9, 5, 35, 16]
    extra_diffusion_steps = EXTRA_DIFFUSION_STEPS if exp_name == "diffusion" else [0]
    use_kdes = [False, True] if exp_name == "diffusion" else [False]
    guide_weight_list = GUIDE_WEIGHTS if exp_name == "cfg" else [None]
    idxs_data = [[] for _ in range(len(idxs))]
    for extra_diffusion_step, guide_weight, use_kde in product(extra_diffusion_steps, guide_weight_list, use_kdes):
        if extra_diffusion_step != 0 and use_kde:
            continue
        for i, idx in enumerate(idxs):
            x_eval = (
                torch.Tensor(torch_data_train.image_all[idx])
                .type(torch.FloatTensor)
                .to(device)
            )
            x_eval_large = torch_data_train.image_all_large[idx]
            obj_mask_eval = torch_data_train.label_all[idx]
            if i == 0:
                obj_mask_eval_marginal = np.zeros_like(obj_mask_eval)
            obj_mask_eval_marginal += obj_mask_eval
            for j in range(6 if not use_kde else 300):
                x_eval_ = x_eval.repeat(50, 1, 1, 1)
                with torch.set_grad_enabled(requires_grad_for_eval):
                    if exp_name == "cfg":
                        model.guide_w = guide_weight
                    if model_type != "diffusion":
                        y_pred_ = model.sample(x_eval_).detach().cpu().numpy()
                    else:
                        if extra_diffusion_step == 0:
                            y_pred_ = (
                                model.sample(x_eval_, extract_embedding=True)
                                .detach()
                                .cpu()
                                .numpy()
                            )

                            if use_kde:
                                # kde
                                torch_obs_many = x_eval_
                                action_pred_many = model.sample(torch_obs_many).cpu().numpy()
                                # fit kde to the sampled actions
                                kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(action_pred_many)
                                # choose the max likelihood one
                                log_density = kde.score_samples(action_pred_many)
                                idx = np.argmax(log_density)
                                y_pred_ = action_pred_many[idx][None, :]
                        else:
                            y_pred_ = model.sample_hack(x_eval_, extra_steps=extra_diffusion_step).detach().cpu().numpy()
                if j == 0:
                    y_pred = y_pred_
                else:
                    y_pred = np.concatenate([y_pred, y_pred_])
            x_eval = x_eval.detach().cpu().numpy()

            idxs_data[i] = {
                "idx": idx,
                "x_eval_large": x_eval_large,
                "obj_mask_eval": obj_mask_eval,
                "y_pred": y_pred,
            }

        # Save data as a pickle
        true_exp_name = exp_name
        if extra_diffusion_step != 0:
            true_exp_name = f"{exp_name}_extra-diffusion_{extra_diffusion_step}"
        if use_kde:
            true_exp_name = f"{exp_name}_kde"
        if guide_weight is not None:
            true_exp_name = f"{exp_name}_guide-weight_{guide_weight}"
        with open(os.path.join(SAVE_DATA_DIR, f"{true_exp_name}.pkl"), "wb") as f:
            pickle.dump(idxs_data, f)


if __name__ == "__main__":
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    for experiment in EXPERIMENTS:
        main(experiment)
