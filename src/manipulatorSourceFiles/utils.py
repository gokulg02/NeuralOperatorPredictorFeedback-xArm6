import matplotlib.pyplot as plt

def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
    
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),#0
     ('dotted',                (0, (1, 5))),#1
     ('densely dotted',        (0, (1, 1))),#2
     ('long dash with offset', (5, (10, 3))),#3
     ('loosely dashed',        (0, (5, 10))),#4
     ('dashed',                (0, (5, 5))),#5
     ('densely dashed',        (0, (5, 1))),#6
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),#7
     ('dashdotted',            (0, (3, 5, 1, 5))),#8
     ('densely dashdotted',    (0, (3, 1, 1, 1))),#9
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),#10
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),#11
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]#12



def plot_trajectory(link, t, nD, qdes, model_states, numerical_states, \
                 model_predictors, numerical_predictors, model_controls, numerical_controls, model_name, saveFig=None):
    fig = plt.figure(figsize=set_size(433, 0.99, (2,2), height_add=0.5))
    subfigs = fig.subfigures(nrows=2, ncols=1, hspace=0)
    subfig = subfigs[0]
    
    subfig.subplots_adjust(left=0.08, bottom=0.2, right=0.98, top=0.85, wspace=0.2, hspace=0.)
    axes = subfig.subplots(nrows=1, ncols=2)
    axes[0].set_title("System state $X(t)$")
    line1, =axes[0].plot(t, qdes[:-5, link])
    line2, =axes[0].plot(t, numerical_states[:, link], color="green", linestyle=linestyle_tuple[2][1])
    line3, =axes[0].plot(t, model_states[:, link], color="orange", linestyle=linestyle_tuple[5][1])
    axes[0].set_xticks([0, 2.5, 5, 7.5, 10])
    
    axes[1].set_title(r"Approximate predictor $\hat{P}(t)$")
    axes[1].plot(t,numerical_predictors[:, -1, link] , color="green",  linestyle=linestyle_tuple[2][1])
    axes[1].plot(t, model_predictors[:, -1, link] , color="orange",  linestyle=linestyle_tuple[5][1])
    
    axes[1].set_xticks([0, 2.5, 5, 7.5, 10])
    axes[0].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    axes[1].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    axes[0].axvline(x=0.5, color='grey', linestyle='--', linewidth=1)
    
    subfig = subfigs[1]
    subfig.subplots_adjust(left=0.08, bottom=0.3, right=0.98, top=0.95, wspace=0.2, hspace=0.)
    
    axes = subfig.subplots(nrows=1, ncols=2)
    axes[0].set_title(r"Control $\kappa(\hat{P}(t))$")
    axes[0].plot(t, numerical_controls[:-5, link], color="green", linestyle=linestyle_tuple[2][1])
    axes[0].plot(t, model_controls[:-5, link], color="orange", linestyle=linestyle_tuple[5][1])
    
    axes[0].set_xlabel("Time t", labelpad=0)
    axes[0].set_xticks([0, 2.5, 5, 7.5, 10])
    axes[1].set_xticks([0, 2.5, 5, 7.5, 10])
    
    axes[1].set_xlabel("Time t", labelpad=0)
    axes[1].set_title(r"Prediction error $\hat{P}(t) - X(t+D)$")
    axes[1].plot(t[:-nD-5], abs(numerical_states[nD+5:, link] - numerical_predictors[nD+1:-4, -1, link]), color="green")
    axes[1].plot(t[:-nD-5], abs(model_states[nD+5:, link] - model_predictors[nD+1:-4,-1, link]), color="orange")
    
    plt.yscale("log")
    fig.legend(handles=[line1, line2, line3],
               labels=["Desired trajectory", "Successive Approximations", model_name],
               loc='lower center',
               bbox_to_anchor=(0.5, -0.015),
               ncol=3,
               fontsize='small')
    if saveFig is not None:
        plt.savefig(saveFig, dpi=300)
    plt.show()

def count_parameters(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
