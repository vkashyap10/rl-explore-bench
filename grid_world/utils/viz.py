import matplotlib.pyplot as plt
import numpy as np


def init_reward_heatmaps(width: int, height: int):
    """Create the reward / variance window with two independent colour‑bars."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.ion()

    imgs = []
    cbars = []
    titles = ["Mean reward  ⟨R⟩", "Reward variance  Var[R]"]

    for ax, title in zip(axes, titles):
        # Use a dummy image with NaNs — no fixed vmin/vmax
        img = ax.imshow(np.full((height, width), np.nan), cmap="magma", origin="upper")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value")
        imgs.append(img)
        cbars.append(cbar)

    return fig, axes, imgs, cbars


def update_reward_heatmaps(imgs, r_mean, r_var, width, height, episode, fig, cbars):
    """Redraw mean & variance maps, interchanging X and Y axes."""

    mean_s = r_mean.mean(axis=1).reshape(width, height).T  # Transpose
    var_s = r_var.mean(axis=1).reshape(width, height).T  # Transpose

    data = [mean_s, var_s]

    for img, cbar, dat in zip(imgs, cbars, data):
        img.set_data(dat)
        img.set_clim(vmin=dat.min(), vmax=dat.max())
        cbar.update_normal(img)

    fig.suptitle(f"Rewards – Episode {episode}", fontsize=14)
    fig.canvas.draw()
    fig.canvas.flush_events()


def init_transition_heatmaps(width, height):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    plt.ion()

    imgs = []
    dots = []
    action_labels = ["0: up", "1: down", "2: left", "3: right"]

    for d, ax in enumerate(axes):
        img = ax.imshow(
            np.zeros((width, height)).T,
            cmap="viridis",
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )
        imgs.append(img)

        ax.set_title(f"Next-state distribution for action {action_labels[d]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        (dot,) = ax.plot([], [], "ro", markersize=6)
        dots.append(dot)

    cbar = fig.colorbar(imgs[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("P(next state)")

    return fig, axes, imgs, dots


def update_transition_heatmaps(
    imgs, probs, state0, width, height, episode, fig, dots=None
):
    """
    Visualize P(s′ | s₀, a) as heatmaps for all actions, and mark state₀ on each plot.
    """
    for action in range(len(imgs)):
        pr_next = probs[:, state0, action]  # shape (num_states,)
        pr_next_grid = pr_next.reshape((width, height)).T  # Transposed

        imgs[action].set_data(pr_next_grid)
        imgs[action].set_clim(0.0, pr_next_grid.max())

        if dots is not None:
            y, x = divmod(state0, width)
            dots[action].set_data([x], [y])

    fig.suptitle(
        f"Transition distribution from state {state0} (X={state0 % width}, Y={state0 // width}) – Episode {episode}",
        fontsize=10,
    )
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_reward_curves(psrl: np.ndarray, vapor: np.ndarray):
    mean_psrl = psrl.mean(axis=0)
    std_psrl = psrl.std(axis=0)
    mean_vapor = vapor.mean(axis=0)
    std_vapor = vapor.std(axis=0)

    episodes = np.arange(len(mean_psrl))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_psrl, label="PSRL", color="blue")
    plt.fill_between(
        episodes, mean_psrl - std_psrl, mean_psrl + std_psrl, alpha=0.2, color="blue"
    )

    plt.plot(episodes, mean_vapor, label="VAPOR", color="green")
    plt.fill_between(
        episodes,
        mean_vapor - std_vapor,
        mean_vapor + std_vapor,
        alpha=0.2,
        color="green",
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Mean ± Std Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_exploration_uncertainty(std_psrl_all: np.ndarray, std_vapor_all: np.ndarray):
    episodes = np.arange(std_psrl_all.shape[1])
    mean_std_psrl = std_psrl_all.mean(0)
    std_std_psrl = std_psrl_all.std(0)
    mean_std_vapor = std_vapor_all.mean(0)
    std_std_vapor = std_vapor_all.std(0)

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, mean_std_psrl, label="PSRL: mean reward std", color="blue")
    plt.fill_between(
        episodes,
        mean_std_psrl - std_std_psrl,
        mean_std_psrl + std_std_psrl,
        alpha=0.2,
        color="blue",
    )

    plt.plot(episodes, mean_std_vapor, label="VAPOR: mean reward std", color="green")
    plt.fill_between(
        episodes,
        mean_std_vapor - std_std_vapor,
        mean_std_vapor + std_std_vapor,
        alpha=0.2,
        color="green",
    )

    plt.xlabel("Episode")
    plt.ylabel("Avg Reward Std (per state-action)")
    plt.title("Exploration Uncertainty Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
