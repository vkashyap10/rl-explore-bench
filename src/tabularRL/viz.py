import matplotlib.pyplot as plt
import numpy as np

def init_value_heatmaps(width, height):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plt.ion()
    imgs = []
    for d, ax in enumerate(axes):
        img = ax.imshow(np.zeros((height, width)),
                        cmap="viridis", origin="upper")
        ax.set_title(f"Orientation {d}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax.invert_yaxis()
        imgs.append(img)

    # one colour‑bar for the last image (all share the same cmap & limits)
    cbar = fig.colorbar(imgs[-1], ax=axes.ravel().tolist(),
                        fraction=0.02, pad=0.02)
    cbar.set_label("State value")
    return fig, axes, imgs


def update_value_heatmaps(imgs, value, width, height, episode, fig,
                          title_prefix="Value Function"):
    v = value.reshape(height, width, 4)
    vmin, vmax = v.min(), v.max()            # keep scale consistent

    for d, img in enumerate(imgs):
        img.set_data(v[:, :, d])
        img.set_clim(vmin, vmax)

    fig.suptitle(f"{title_prefix} – Episode {episode}", fontsize=14)
    fig.canvas.draw()
    fig.canvas.flush_events()


# ────────────────────────────────────────────────────────────────
#   FIGURE 2 – rewards (mean & variance)
# ────────────────────────────────────────────────────────────────

def init_reward_heatmaps(width: int, height: int):
    """Create the reward / variance window with two independent colour‑bars."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.ion()

    imgs = []
    cbars = []
    titles = ["Mean reward  ⟨R⟩", "Reward variance  Var[R]"]

    for ax, title in zip(axes, titles):
        img = ax.imshow(np.zeros((height, width)),
                        cmap="magma", origin="upper")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # a *dedicated* colour‑bar for this axis
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value")
        imgs.append(img)
        cbars.append(cbar)

    return fig, axes, imgs, cbars


def update_reward_heatmaps(imgs, r_mean, r_var, width, height,
                           episode, fig, cbars):
    """Redraw mean & variance maps, adjusting each colour‑bar separately."""
    mean_s = r_mean.mean(axis=1).reshape(height, width, 4).mean(axis=2)
    var_s  =  r_var.mean(axis=1).reshape(height, width, 4).mean(axis=2)

    data = [mean_s, var_s]

    for img, cbar, dat in zip(imgs, cbars, data):
        img.set_data(dat)
        # autoscale each map independently
        img.set_clim(vmin=dat.min(), vmax=dat.max())
        cbar.update_normal(img)          # refresh the colour‑bar ticks

    fig.suptitle(f"Rewards – Episode {episode}", fontsize=14)
    fig.canvas.draw()
    fig.canvas.flush_events()


# ────────────────────────────────────────────────────────────────
#   FIGURE 3 – transition probabilities from one chosen state
# ────────────────────────────────────────────────────────────────
def init_transition_heatmaps(width, height):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plt.ion()
    imgs = [ax.imshow(np.zeros((height, width)),
                      cmap="viridis", origin="upper")
            for ax in axes]

    for d, ax in enumerate(axes):
        ax.set_title(f"Next‑state orientation {d}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    cbar = fig.colorbar(imgs[-1], ax=axes.ravel().tolist(),
                        fraction=0.02, pad=0.02)
    cbar.set_label("Pr(next state)")
    return fig, axes, imgs


def update_transition_heatmaps(imgs, probs, state0, width, height, episode, fig):
    """probs : shape (S,S,A)  — use *sampled* P from this episode."""
    # aggregate over actions → Pr(s′ | s₀)  (shape S,)
    pr_next = probs[:, state0, :].mean(axis=1)         # (S,)
    pr_next = pr_next.reshape(height, width, 4)

    vmin, vmax = 0.0, pr_next.max()
    for d, img in enumerate(imgs):
        img.set_data(pr_next[:, :, d])
        img.set_clim(vmin, vmax)

    fig.suptitle(f"Transition probs from state {state0} – Episode {episode}",
                 fontsize=14)
    fig.canvas.draw()
    fig.canvas.flush_events()