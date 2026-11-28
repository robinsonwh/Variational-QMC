import numpy as np

def projected_density(x, y, L=4, bins=150, n_electrons=1):
    """2D number density of (x, y) points, normalised to integrate to n_electrons."""
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-L, L], [-L, L]],
                                       density=True)
    return n_electrons * H, xedges, yedges

def plot_density(ax, H, xedges, yedges, title, vmax=None, cmap="inferno"):
    im = ax.pcolormesh(xedges, yedges, H.T, vmin=0, vmax=vmax, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    return im