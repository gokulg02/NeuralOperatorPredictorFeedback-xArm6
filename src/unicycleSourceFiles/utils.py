import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def plot_trajectory(t, u_delay, u_delay_ml, control_delay, control_delay_ml, savefig=None):
    fig = plt.figure(figsize=set_size(433, 0.99, (3, 3), height_add=0))
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    ax4 = fig.add_subplot(gs[1, 0:3])  # Spans columns 0 and 1
    ax5 = fig.add_subplot(gs[1, 3:6])    # Just column 2
    

    style1 = {'color': 'tab:green', 'linestyle': '-', 'linewidth': 2}
    style2 = {'color': 'tab:orange', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7}

    ax1.plot(t, u_delay[:, 0], label="Const delay", **style1)
    ax1.plot(t, u_delay_ml[:, 0], label="Const delay ML", **style2)
    ax1.set_ylabel(r"$x(t)$")
    ax1.set_xlabel("time")
    ax1.set_xticks([0, 2.5, 5, 7.5, 10])
    ax1.set_yticks([1, 0.5, 0, -0.5, -1])

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_ml[:, 1], **style2)
    ax2.set_ylabel(r"$y(t)$")
    ax2.set_xlabel("time")
    ax2.set_xticks([0, 2.5, 5, 7.5, 10])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_ml[:, 2], **style2)
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"$\theta(t)$")
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_xticks([0, 2.5, 5, 7.5, 10])

    ax4.plot(t, control_delay[:, 0], **style1)
    ax4.plot(t, control_delay_ml[:, 0], **style2)
    ax4.set_xlabel("time")
    ax4.set_ylabel(r"$\nu_1(t)$")
    ax4.set_yticks([-2, -1, 0, 1, 2])
    ax4.set_xticks([0, 2.5, 5, 7.5, 10])

    l1, = ax5.plot(t, control_delay[:, 1], label="Const delay", **style1)
    l2, = ax5.plot(t, control_delay_ml[:, 1], label="Const delay ML", **style2)
    ax5.set_xlabel("time")
    ax5.set_ylabel(r"$\nu_2(t)$")
    ax5.set_yticks([-2, -1, 0, 1, 2])
    ax5.set_xticks([0, 2.5, 5, 7.5, 10])



    # Add a horizontal legend below all subplots
    fig.legend(handles=[l1, l2],loc='lower center',ncol=3,fontsize=10, frameon=True,fancybox=True, shadow=False,bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    plt.show()
    
def plot_trajectory_with_errors(t, u_delay, u_delay_ml, control_delay, control_delay_ml, predictor, predictor_ml, nD, savefig=None):
    fig = plt.figure(figsize=set_size(433, 0.99, (3, 3), height_add=1))
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1])  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    ax4 = fig.add_subplot(gs[1, 0:3])  # Spans columns 0 and 1
    ax5 = fig.add_subplot(gs[1, 3:6])    # Just column 2
    
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax7 = fig.add_subplot(gs[2, 2:4])
    ax8 = fig.add_subplot(gs[2, 4:6])
    

    style1 = {'color': 'tab:green', 'linestyle': '-', 'linewidth': 2}
    style2 = {'color': 'tab:orange', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7}

    ax1.plot(t, u_delay[:, 0], label="Const delay", **style1)
    ax1.plot(t, u_delay_ml[:, 0], label="Const delay ML", **style2)
    ax1.set_ylabel(r"$x(t)$")
    ax1.set_xlabel("time")
    ax1.set_xticks([0, 2.5, 5, 7.5, 10])
    ax1.set_yticks([1, 0.5, 0, -0.5, -1])

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_ml[:, 1], **style2)
    ax2.set_ylabel(r"$y(t)$")
    ax2.set_xlabel("time")
    ax2.set_xticks([0, 2.5, 5, 7.5, 10])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_ml[:, 2], **style2)
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"$\theta(t)$")
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_xticks([0, 2.5, 5, 7.5, 10])

    ax4.plot(t, control_delay[:, 0], **style1)
    ax4.plot(t, control_delay_ml[:, 0], **style2)
    ax4.set_xlabel("time")
    ax4.set_ylabel(r"$\nu_1(t)$")
    ax4.set_yticks([-2, -1, 0, 1, 2])
    ax4.set_xticks([0, 2.5, 5, 7.5, 10])

    l1, = ax5.plot(t, control_delay[:, 1], label="Const delay", **style1)
    l2, = ax5.plot(t, control_delay_ml[:, 1], label="Const delay ML", **style2)
    ax5.set_xlabel("time")
    ax5.set_ylabel(r"$\nu_2(t)$")
    ax5.set_yticks([-2, -1, 0, 1, 2])
    ax5.set_xticks([0, 2.5, 5, 7.5, 10])

    ax6.plot(t[0:-nD+2], abs(predictor[0:-nD+2, -1, 0]-u_delay[nD-2:, 0]), label="Const delay", **style1)
    ax6.plot(t[0:-nD+2], abs(predictor_ml[0:-nD+2,-1, 0]-u_delay[nD-2:, 0]), **style2)
    ax6.set_ylabel(r"$|x(t+D)-\hat{P}(t)|$")
    ax6.set_xlabel("time")
    ax6.set_xticks([0, 2.5, 5, 7.5, 10])
    ax6.set_yticks([0, 0.05, 0.1])

    ax7.plot(t[0:-nD+2], abs(predictor[0:-nD+2, -1, 1]-u_delay[nD-2:, 1]), label="Const delay", **style1)
    ax7.plot(t[0:-nD+2], abs(predictor_ml[0:-nD+2,-1, 1]-u_delay[nD-2:, 1]), **style2)
    ax7.set_ylabel(r"$|y(t+D)-\hat{P}(t)|$")
    ax7.set_xlabel("time")
    ax7.set_xticks([0, 2.5, 5, 7.5, 10])
    ax7.set_yticks([0, 0.05, 0.1])

    ax8.plot(t[0:-nD+2], abs(predictor[0:-nD+2, -1, 2]-u_delay[nD-2:, 2]), label="Const delay", **style1)
    ax8.plot(t[0:-nD+2], abs(predictor_ml[0:-nD+2,-1, 2]-u_delay[nD-2:, 2]), **style2)
    ax8.set_xlabel("time")
    ax8.set_ylabel(r"$|\theta(t+D)-\hat{P}(t)|$")
    ax8.set_xticks([0, 2.5, 5, 7.5, 10])
    ax8.set_yticks([0, 0.05, 0.1])

    # Add a horizontal legend below all subplots
    fig.legend(handles=[l1, l2],loc='lower center',ncol=3,fontsize=10, frameon=True,fancybox=True, shadow=False,bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    plt.show()

def count_parameters(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
