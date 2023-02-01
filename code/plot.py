# If you are experiencing errors with latex rendering, try installing the following (on Ubuntu):
# sudo apt install texlive texlive-latex-extra texlive-latex-recommended dvipng cm-super msttcorefonts

import os
import pickle

import numpy as np
from matplotlib import pyplot

SAVE_DATA_DIR = "output"
SAVE_FIGURE_DIR = "figures"
DPI = 300

TITLE_KWARGS = {"fontsize": 7, "multialignment": "center", "verticalalignment": "center", "horizontalalignment": "center"}
YLABEL_TITLE_KWARGS = {"fontsize": "small", "verticalalignment": "center", "horizontalalignment": "right", "rotation": 0}
SAVEFIG_KWARGS = {"dpi": DPI, "bbox_inches": "tight", "pad_inches": 0.0}
SCATTER_KWARGS = {"s": 1, "alpha": 0.25, "color": "yellow"}

MAIN_PLOT_EXPERIMENTS = [
    {
        "exp_name": "mse",
        "clean_name": "MSE",
    },
    {
        "exp_name": "meanvariance",
        "clean_name": "Gaussian",
    },
    {
        "exp_name": "discrete",
        "clean_name": "Discretised",
    },
    {
        "exp_name": "kmeans",
        "clean_name": "K-Means",
    },
    {
        "exp_name": "bet",
        "clean_name": "K-Means$+$Res.",
    },
    {
        "exp_name": "ebm_derivative_free",
        "clean_name": "EBM Deriv.-Free",
    },
    {
        "exp_name": "ebm_langevin",
        "clean_name": "EBM Langevin",
    },
    {
        "exp_name": "diffusion",
        "clean_name": "Diffusion",
    },
]
N_SAMPLES_PER_METHOD = 7
INTERESTING_SAMPLE_IDX = 6
EXTRA_DIFFUSION_STEPS = [0, 2, 4, 8, 16, 32]
GUIDE_WEIGHTS = [0.0, 4.0, 8.0]
MASK_WEIGHT = 0.3

pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


def plot_main_comparison():
    MY_DIFFUSION_STEPS = [4, 16, 32]
    fig, axs = pyplot.subplots(
        nrows=len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS) + 1 + 2,
        ncols=N_SAMPLES_PER_METHOD,
        dpi=DPI
    )
    ylabel_title_kwargs = YLABEL_TITLE_KWARGS.copy()
    ylabel_title_kwargs["fontsize"] = 7
    for experiment_i, experiment in enumerate(MAIN_PLOT_EXPERIMENTS):
        file_path = os.path.join(SAVE_DATA_DIR, experiment["exp_name"] + ".pkl")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for sample_i in range(N_SAMPLES_PER_METHOD):
            # If first experiment, use base images from it
            if experiment_i == 0:
                axs[0, sample_i].imshow(data[sample_i]["x_eval_large"])
                axs[1, sample_i].imshow(data[sample_i]["obj_mask_eval"])
                if sample_i == 0:
                    axs[0, sample_i].set_ylabel("Observation $\mathbf{o}$", **ylabel_title_kwargs)
                    axs[1, sample_i].set_ylabel("$p(\mathbf{a} | \mathbf{o})$", **ylabel_title_kwargs)
                for z in range(2):
                    axs[z, sample_i].set_xticks([])
                    axs[z, sample_i].set_yticks([])
                    axs[z, sample_i].set_ylim(63, -1)
                    axs[z, sample_i].set_xlim(63, -1)
            axs[2 + experiment_i, sample_i].imshow(
                data[sample_i]["obj_mask_eval"] * MASK_WEIGHT,
                vmin=0,
                vmax=1
            )
            axs[2 + experiment_i, sample_i].scatter(
                data[sample_i]["y_pred"][:, 0] * 64,
                data[sample_i]["y_pred"][:, 1] * 64,
                **SCATTER_KWARGS
            )
            axs[2 + experiment_i, sample_i].set_xticks([])
            axs[2 + experiment_i, sample_i].set_yticks([])
            axs[2 + experiment_i, sample_i].set_ylim(63, -1)
            axs[2 + experiment_i, sample_i].set_xlim(63, -1)
            if sample_i == 0:
                axs[2 + experiment_i, sample_i].set_ylabel(experiment["clean_name"], **ylabel_title_kwargs)
    # Plot the extra-diffusion steps
    for diffusion_step_i, diffusion_step in enumerate(MY_DIFFUSION_STEPS):
        file_path = os.path.join(SAVE_DATA_DIR, "diffusion_extra-diffusion_{}.pkl".format(diffusion_step))
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for sample_i in range(N_SAMPLES_PER_METHOD):
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].imshow(
                data[sample_i]["obj_mask_eval"] * MASK_WEIGHT,
                vmin=0,
                vmax=1
            )
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].scatter(
                data[sample_i]["y_pred"][:, 0] * 64,
                data[sample_i]["y_pred"][:, 1] * 64,
                **SCATTER_KWARGS
            )
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].set_xticks([])
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].set_yticks([])
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].set_ylim(63, -1)
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].set_xlim(63, -1)
            if sample_i == 0:
                axs[2 + len(MAIN_PLOT_EXPERIMENTS) + diffusion_step_i, sample_i].set_ylabel(
                    "Diff.X. ($M={}$)".format(diffusion_step),
                    **ylabel_title_kwargs
                )
    # Plot the kde
    file_path = os.path.join(SAVE_DATA_DIR, "diffusion_kde.pkl")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    for sample_i in range(N_SAMPLES_PER_METHOD):
        axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].imshow(
            data[sample_i]["obj_mask_eval"] * MASK_WEIGHT,
            vmin=0,
            vmax=1
        )
        axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].scatter(
            data[sample_i]["y_pred"][:, 0] * 64,
            data[sample_i]["y_pred"][:, 1] * 64,
            **SCATTER_KWARGS
        )
        axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].set_xticks([])
        axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].set_yticks([])
        axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].set_ylim(63, -1)
        axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].set_xlim(63, -1)
        if sample_i == 0:
            axs[2 + len(MAIN_PLOT_EXPERIMENTS) + len(MY_DIFFUSION_STEPS), sample_i].set_ylabel(
                "Diff. KDE",
                **ylabel_title_kwargs
            )

    # Save fig
    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=-0.85, hspace=0.1)
    fig.savefig(os.path.join(SAVE_FIGURE_DIR, "main_comparison.pdf"), **SAVEFIG_KWARGS)


def plot_cfg_comparison():
    fig, axs = pyplot.subplots(nrows=len(GUIDE_WEIGHTS) + 2, ncols=N_SAMPLES_PER_METHOD, dpi=DPI)
    all_masks = []
    for guide_weight_i, guide_weight in enumerate(GUIDE_WEIGHTS):
        file_path = os.path.join(SAVE_DATA_DIR, "cfg_guide-weight_{}.pkl".format(guide_weight))
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for sample_i in range(N_SAMPLES_PER_METHOD):
            # If first experiment, use base images from it
            if guide_weight_i == 0:
                axs[0, sample_i].imshow(data[sample_i]["obj_mask_eval"])
                for z in range(1):
                    axs[z, sample_i].set_xticks([])
                    axs[z, sample_i].set_yticks([])
                    axs[z, sample_i].set_ylim(63, -1)
                    axs[z, sample_i].set_xlim(63, -1)
                if sample_i == 0:
                    # Titles
                    axs[0, sample_i].set_ylabel("$p(\mathbf{a} | \mathbf{o})$", **YLABEL_TITLE_KWARGS)
            axs[2 + guide_weight_i, sample_i].imshow(
                data[sample_i]["obj_mask_eval"] * MASK_WEIGHT,
                vmin=0,
                vmax=1
            )
            if guide_weight_i == 0:
                all_masks.append(data[sample_i]["obj_mask_eval"])
            axs[2 + guide_weight_i, sample_i].scatter(
                data[sample_i]["y_pred"][:, 0] * 64,
                data[sample_i]["y_pred"][:, 1] * 64,
                **SCATTER_KWARGS
            )
            axs[2 + guide_weight_i, sample_i].set_xticks([])
            axs[2 + guide_weight_i, sample_i].set_yticks([])
            axs[2 + guide_weight_i, sample_i].set_ylim(63, -1)
            axs[2 + guide_weight_i, sample_i].set_xlim(63, -1)
            if sample_i == 0:
                # Titles
                axs[2 + guide_weight_i, sample_i].set_ylabel("Weight $= " + str(guide_weight) + "$", **YLABEL_TITLE_KWARGS)
    # Create the "unique" parts image,
    # where it is "mask - union_of_all_masks"
    all_masks = np.array(all_masks)
    for sample_i in range(N_SAMPLES_PER_METHOD):
        other_masks = np.concatenate([all_masks[:sample_i], all_masks[sample_i + 1:]])
        axs[1, sample_i].imshow(
            all_masks[sample_i] - np.max(other_masks, axis=0),
            vmin=0,
            vmax=1
        )
        axs[1, sample_i].set_xticks([])
        axs[1, sample_i].set_yticks([])
        axs[1, sample_i].set_ylim(63, -1)
        axs[1, sample_i].set_xlim(63, -1)
        if sample_i == 0:
            # Titles
            axs[1, sample_i].set_ylabel("Unique $p(\mathbf{a}|\mathbf{o})$", **YLABEL_TITLE_KWARGS)

    # Save fig
    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=0.2, hspace=-0.6)
    fig.savefig(os.path.join(SAVE_FIGURE_DIR, "cfg_comparison.pdf"), **SAVEFIG_KWARGS)


def plot_extra_diff_comparison():
    fig, axs = pyplot.subplots(nrows=1, ncols=len(EXTRA_DIFFUSION_STEPS) + 3, dpi=DPI)
    # Plot different diffusion steps for the interesting sample
    for diffusion_i, diffusion_step in enumerate(EXTRA_DIFFUSION_STEPS):
        if diffusion_step == 0:
            file_path = os.path.join(SAVE_DATA_DIR, "diffusion.pkl")
        else:
            file_path = os.path.join(SAVE_DATA_DIR, "diffusion_extra-diffusion_{}.pkl".format(diffusion_step))
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if diffusion_step == 0:
            axs[0].imshow(data[INTERESTING_SAMPLE_IDX]["x_eval_large"])
            axs[1].imshow(data[INTERESTING_SAMPLE_IDX]["obj_mask_eval"])
            axs[0].set_title("Obs. $\mathbf{o}$", **TITLE_KWARGS)
            axs[1].set_title("$p(\mathbf{a} | \mathbf{o})$", **TITLE_KWARGS)
            for z in range(2):
                axs[z].set_xticks([])
                axs[z].set_yticks([])
                axs[z].set_ylim(63, -1)
                axs[z].set_xlim(63, -1)
        axs[diffusion_i + 2].imshow(
            data[INTERESTING_SAMPLE_IDX]["obj_mask_eval"] * MASK_WEIGHT,
            vmin=0,
            vmax=1
        )
        axs[diffusion_i + 2].scatter(
            data[INTERESTING_SAMPLE_IDX]["y_pred"][:, 0] * 64,
            data[INTERESTING_SAMPLE_IDX]["y_pred"][:, 1] * 64,
            **SCATTER_KWARGS
        )
        axs[diffusion_i + 2].set_xticks([])
        axs[diffusion_i + 2].set_yticks([])
        axs[diffusion_i + 2].set_ylim(63, -1)
        axs[diffusion_i + 2].set_xlim(63, -1)
        if diffusion_step == 0:
            axs[diffusion_i + 2].set_title("$M=0$", **TITLE_KWARGS)
        else:
            axs[diffusion_i + 2].set_title("$M={}$".format(diffusion_step), **TITLE_KWARGS)

    # Plot KDE diffusion result
    with open(os.path.join(SAVE_DATA_DIR, "diffusion_kde.pkl"), "rb") as f:
        data = pickle.load(f)
    axs[-1].imshow(
        data[INTERESTING_SAMPLE_IDX]["obj_mask_eval"] * MASK_WEIGHT,
        vmin=0,
        vmax=1
    )
    axs[-1].scatter(
        data[INTERESTING_SAMPLE_IDX]["y_pred"][:, 0] * 64,
        data[INTERESTING_SAMPLE_IDX]["y_pred"][:, 1] * 64,
        **SCATTER_KWARGS
    )
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])
    axs[-1].set_ylim(63, -1)
    axs[-1].set_xlim(63, -1)
    axs[-1].set_title("Diff. KDE", **TITLE_KWARGS)
    # Save fig
    fig.savefig(os.path.join(SAVE_FIGURE_DIR, "extra_diffusion_comparison.pdf"), **SAVEFIG_KWARGS)


def plot_intro_figure():
    filtered_plot_experiments = [experiment for experiment in MAIN_PLOT_EXPERIMENTS if "ebm" not in experiment["exp_name"]]
    fig, axs = pyplot.subplots(nrows=1, ncols=len(filtered_plot_experiments) + 2, dpi=DPI)
    title_kwargs = TITLE_KWARGS.copy()
    title_kwargs["fontsize"] = 7
    # Plot interesting sample index for each experiment
    for experiment_i, experiment in enumerate(filtered_plot_experiments):
        # Use t+16 diffusion for the diffusion
        if experiment["exp_name"] == "diffusion":
            file_path = os.path.join(SAVE_DATA_DIR, "diffusion_extra-diffusion_16.pkl")
        else:
            file_path = os.path.join(SAVE_DATA_DIR, "{}.pkl".format(experiment["exp_name"]))
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        # Show main image
        axs[0].imshow(data[INTERESTING_SAMPLE_IDX]["x_eval_large"])
        axs[0].set_title("Obs. $\mathbf{o}$", **title_kwargs)
        axs[1].imshow(data[INTERESTING_SAMPLE_IDX]["obj_mask_eval"])
        axs[1].set_title("$p(\mathbf{a} | \mathbf{o})$", **title_kwargs)
        for z in range(2):
            axs[z].set_xticks([])
            axs[z].set_yticks([])
            axs[z].set_ylim(64, 0)
            axs[z].set_xlim(64, 0)
        axs[experiment_i + 2].imshow(
            data[INTERESTING_SAMPLE_IDX]["obj_mask_eval"] * MASK_WEIGHT,
            vmin=0,
            vmax=1
        )
        axs[experiment_i + 2].scatter(
            data[INTERESTING_SAMPLE_IDX]["y_pred"][:, 0] * 64,
            data[INTERESTING_SAMPLE_IDX]["y_pred"][:, 1] * 64,
            **SCATTER_KWARGS
        )
        axs[experiment_i + 2].set_title("{}".format(experiment["clean_name"]), **title_kwargs)
        axs[experiment_i + 2].set_xticks([])
        axs[experiment_i + 2].set_yticks([])
        axs[experiment_i + 2].set_ylim(63, -1)
        axs[experiment_i + 2].set_xlim(63, -1)
    # Save fig
    # Reduce horizontal spacing
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(SAVE_FIGURE_DIR, "intro_figure.pdf"), **SAVEFIG_KWARGS)


if __name__ == "__main__":
    os.makedirs(SAVE_FIGURE_DIR, exist_ok=True)
    plot_main_comparison()
    plot_cfg_comparison()
    plot_extra_diff_comparison()
    plot_intro_figure()
