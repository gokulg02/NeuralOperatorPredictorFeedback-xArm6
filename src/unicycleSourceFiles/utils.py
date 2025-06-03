import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
     
     


def plot_trajectory_time_varying(t, u_delay, u_delay_ml, control_delay, control_delay_ml, predictors, predictors_ml, savefig=None, axis=None):
    fig = plt.figure(figsize=set_size(516, 0.99, (3, 2), height_add=0))
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1])  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax6 = fig.add_subplot(gs[1, 4:6])

    ax7 = fig.add_subplot(gs[2, 0:3])  # Spans columns 0 and 1
    ax8 = fig.add_subplot(gs[2, 3:6])    # Just column 2
    

    style1 = {'color': 'tab:green', 'linestyle': linestyle_tuple[2][1], 'linewidth': 2}
    style2 = {'color': 'tab:orange', 'linestyle': linestyle_tuple[5][1], 'linewidth': 2, 'alpha': 0.7}

    ax1.plot(t, u_delay[:, 0], label="Const delay", **style1)
    ax1.plot(t, u_delay_ml[:, 0], label="Const delay ML", **style2)
    ax1.set_ylabel(r"$x(t)$")
    ax1.set_xlabel("t")
    ax1.set_xticks([0, 2.5, 5, 7.5, 10])
    ax1.set_yticks([1, 0.5, 0, -0.5, -1])

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_ml[:, 1], **style2)
    ax2.set_ylabel(r"$y(t)$", labelpad=-3)
    ax2.set_xlabel("t")
    ax2.set_xticks([0, 2.5, 5, 7.5, 10])
    ax2.set_yticks([-0.5, 0, 0.5, 1])
    
    if axis: 
        # Create inset axes (zoom factor = 2)
        axins = inset_axes(ax2, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),  # (x0, y0, width, height)
                       bbox_transform=ax2.transAxes,
                       borderpad=0)  # Location of inset
        axins.plot(t, u_delay[:, 1], **style1)
        axins.plot(t, u_delay_ml[:, 1], **style2)
        

        # Limit the region shown in inset
        x1, x2 = 5.5, 6   # x-range for zoom
        y1, y2 = -0.1, -0.05  # y-range for zoom
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # Hide tick labels for the inset
        axins.set_xticks([])
        axins.set_yticks([])

        # Draw lines connecting inset to main plot
        mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_ml[:, 2], **style2)
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\theta(t)$", labelpad=-1)
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_xticks([0, 2.5, 5, 7.5, 10])
    
    ax4.plot(t, predictors[:, -1, 0], label="Const delay", **style1)
    ax4.plot(t, predictors_ml[:, -1, 0], label="Const delay ML", **style2)
    ax4.set_ylabel(r"$P_1(t) \approx x(t+D(t))$")
    ax4.set_xlabel("t")
    ax4.set_xticks([0, 2.5, 5, 7.5, 10])
    ax4.set_yticks([1, 0.5, 0, -0.5, -1])

    
    ax5.plot(t, predictors[:, -1, 1], label="Const delay", **style1)
    ax5.plot(t, predictors_ml[:, -1, 1], label="Const delay ML", **style2)
    ax5.set_ylabel(r"$P_2(t) \approx y(t+D(t))$", labelpad=-3)
    ax5.set_xlabel("t")
    ax5.set_xticks([0, 2.5, 5, 7.5, 10])
    ax5.set_yticks([-0.5, 0, 0.5, 1])


    if axis:
            # Create inset axes (zoom factor = 2)
        axins2 = inset_axes(ax5, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),  # (x0, y0, width, height)
                       bbox_transform=ax5.transAxes,
                       borderpad=0)  # Location of inset
        axins2.plot(t, predictors[:, -1, 1], label="Const delay", **style1)
        axins2.plot(t, predictors_ml[:, -1, 1], label="Const delay ML", **style2)
        

        # Limit the region shown in inset
        x1, x2 = 5.5, 6   # x-range for zoom
        y1, y2 = 0., 0.1  # y-range for zoom
        axins2.set_xlim(x1, x2)
        axins2.set_ylim(y1, y2)

        # Hide tick labels for the inset
        axins2.set_xticks([])
        axins2.set_yticks([])
        mark_inset(ax5, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax6.plot(t, predictors[:, -1, 2], label="Const delay", **style1)
    ax6.plot(t, predictors_ml[:, -1, 2], label="Const delay ML", **style2)
    ax6.set_xlabel("t")
    ax6.set_ylabel(r"$P_3(t) \approx \theta(t+D(t))$", labelpad=-1)
    ax6.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax6.set_xticks([0, 2.5, 5, 7.5, 10])
    plt.subplots_adjust(hspace=0.5, left=0.1, right=0.98, top=0.95, bottom=0.14, wspace=1)

    ax7.plot(t, control_delay[:, 0], **style1)
    ax7.plot(t, control_delay_ml[:, 0], **style2)
    ax7.set_xlabel("t")
    ax7.set_ylabel(r"$\nu_1(t)$")
    ax7.set_yticks([-3, -1.5, 0, 1.5])
    ax7.set_xticks([0, 2.5, 5, 7.5, 10])

    l1, = ax8.plot(t, control_delay[:, 1], label="Successive Approximations", **style1)
    l2, = ax8.plot(t, control_delay_ml[:, 1], label="ML", **style2)
    ax8.set_xlabel("t")
    ax8.set_ylabel(r"$\nu_2(t)$", labelpad=-3)
    ax8.set_yticks([-3, -2, -1,0,1])
    ax8.set_xticks([0, 2.5, 5, 7.5, 10])
    fig.text(0.5, 0.98, "System states", va='center', ha='center', fontsize=16)
    fig.text(0.5, 0.66, "Predictions", va='center', ha='center', fontsize=16)
    fig.text(0.5, 0.37, "Control Inputs", va='center', ha='center', fontsize=16)

    # Add a horizontal legend below all subplots
    fig.legend(handles=[l1, l2],loc='lower center',ncol=3,fontsize=10, frameon=True,fancybox=True, shadow=False,bbox_to_anchor=(0.5, 0.02))
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    plt.show()
    


def plot_trajectory_time_varying_extra_delay(t, u_delay, u_delay_ml, u_delay_ml_2, control_delay, control_delay_ml, control_delay_ml_2, predictors, predictors_ml,predictors_ml_2, phi1, phi2, savefig=None):
    fig = plt.figure(figsize=set_size(516, 0.99, (3, 2), height_add=0))
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1])  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax6 = fig.add_subplot(gs[1, 4:6])

    ax7 = fig.add_subplot(gs[2, 0:2])  # Spans columns 0 and 1
    ax8 = fig.add_subplot(gs[2, 2:4])   
    ax9 = fig.add_subplot(gs[2, 4:6])   
    

    style1 = {'color': 'tab:blue', 'linewidth': 2}
    style2 = {'color': 'tab:orange', 'linestyle': linestyle_tuple[5][1], 'linewidth': 2, 'alpha': 0.7}
    style3 = {'color': 'tab:green', 'linestyle': linestyle_tuple[2][1], 'linewidth': 2, 'alpha': 0.7}

    ax1.plot(t, u_delay[:, 0], label="Const delay", **style1)
    ax1.plot(t, u_delay_ml[:, 0], label="Const delay ML", **style2)
    ax1.plot(t, u_delay_ml_2[:, 0], label="Long delay ML", **style3)
    ax1.set_ylabel(r"$x(t)$")
    ax1.set_xlabel("t")
    ax1.set_xticks([0, 2.5, 5, 7.5, 10])
    ax1.set_yticks([1, 0.5, 0, -0.5, -1])

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_ml[:, 1], **style2)
    ax2.plot(t, u_delay_ml_2[:, 1], **style3)
    ax2.set_ylabel(r"$y(t)$", labelpad=-3)
    ax2.set_xlabel("t")
    ax2.set_xticks([0, 2.5, 5, 7.5, 10])
    ax2.set_yticks([-0.5, 0, 0.5, 1])
    
    
    # Create inset axes (zoom factor = 2)
    axins = inset_axes(ax2, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),  # (x0, y0, width, height)
                   bbox_transform=ax2.transAxes,
                   borderpad=0)  # Location of inset
    axins.plot(t, u_delay[:, 1], **style1)
    axins.plot(t, u_delay_ml[:, 1], **style2)
    axins.plot(t, u_delay_ml_2[:, 1], **style3)
    

    # Limit the region shown in inset
    x1, x2 = 5.5, 6   # x-range for zoom
    y1, y2 = -0.1, 0.05  # y-range for zoom
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # Hide tick labels for the inset
    axins.set_xticks([])
    axins.set_yticks([])

    # Draw lines connecting inset to main plot
    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_ml[:, 2], **style2)
    ax3.plot(t, u_delay_ml_2[:, 2], **style3)
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\theta(t)$", labelpad=-1)
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_xticks([0, 2.5, 5, 7.5, 10])
    
    ax4.plot(t, predictors[:, -1, 0], label="Const delay", **style1)
    ax4.plot(t, predictors_ml[:, -1, 0], label="Const delay ML", **style2)
    ax4.plot(t, predictors_ml_2[:, -1, 0], label="Const delay ML", **style3)
    ax4.set_ylabel(r"$P_1(t) \approx x(t+D(t))$")
    ax4.set_xlabel("t")
    ax4.set_xticks([0, 2.5, 5, 7.5, 10])
    ax4.set_yticks([1, 0.5, 0, -0.5, -1])

    
    ax5.plot(t, predictors[:, -1, 1], label="Const delay", **style1)
    ax5.plot(t, predictors_ml[:, -1, 1], label="Const delay ML", **style2)
    ax5.plot(t, predictors_ml_2[:, -1, 1], label="Const delay ML", **style3)
    ax5.set_ylabel(r"$P_2(t) \approx y(t+D(t))$", labelpad=-3)
    ax5.set_xlabel("t")
    ax5.set_xticks([0, 2.5, 5, 7.5, 10])
    ax5.set_yticks([-0.5, 0, 0.5, 1])
    
        # Create inset axes (zoom factor = 2)
    axins2 = inset_axes(ax5, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),  # (x0, y0, width, height)
                   bbox_transform=ax5.transAxes,
                   borderpad=0)  # Location of inset
    axins2.plot(t, predictors[:, -1, 1], label="Const delay", **style1)
    axins2.plot(t, predictors_ml[:, -1, 1], label="Const delay ML", **style2)
    axins2.plot(t, predictors_ml_2[:, -1, 1], label="Const delay ML", **style3)
    # Limit the region shown in inset
    x1, x2 = 5.5, 6   # x-range for zoom
    y1, y2 = -0.1, 0.05  # y-range for zoom
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)

    # Hide tick labels for the inset
    axins2.set_xticks([])
    axins2.set_yticks([])
    mark_inset(ax5, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax6.plot(t, predictors[:, -1, 2], label="Const delay", **style1)
    ax6.plot(t, predictors_ml[:, -1, 2], label="Const delay ML", **style2)
    ax6.plot(t, predictors_ml_2[:, -1, 2], label="Const delay ML", **style3)
    ax6.set_xlabel("t")
    ax6.set_ylabel(r"$P_3(t) \approx \theta(t+D(t))$", labelpad=0)
    ax6.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax6.set_xticks([0, 2.5, 5, 7.5, 10])
    plt.subplots_adjust(hspace=0.5, left=0.1, right=0.98, top=0.95, bottom=0.14, wspace=1)

    ax7.plot(t, control_delay[:, 0], **style1)
    ax7.plot(t, control_delay_ml[:, 0], **style2)
    ax7.plot(t, control_delay_ml_2[:, 0], **style3)
    ax7.set_xlabel("t")
    ax7.set_ylabel(r"$\nu_1(t)$")
    ax7.set_yticks([-3, -1.5, 0, 1.5])
    ax7.set_xticks([0, 2.5, 5, 7.5, 10])

    l1, = ax8.plot(t, control_delay[:, 1], label="Numerical Method", **style1)
    l2, = ax8.plot(t, control_delay_ml[:, 1], label="FNO $\phi_1(t)$", **style2)
    l3, = ax8.plot(t, control_delay_ml_2[:, 1], label="FNO $\phi_2(t)$", **style3)
    ax8.set_xlabel("t")
    ax8.set_ylabel(r"$\nu_2(t)$", labelpad=-3)
    ax8.set_yticks([-3, -2, -1,0,1])
    ax8.set_xticks([0, 2.5, 5, 7.5, 10])
    fig.text(0.5, 0.98, "System states", va='center', ha='center', fontsize=16)
    fig.text(0.5, 0.66, "Predictions", va='center', ha='center', fontsize=16)
    fig.text(0.36, 0.36, "Control Inputs", va='center', ha='center', fontsize=16)
    fig.text(0.87, 0.36, "Delay functions", va='center', ha='center', fontsize=16)
    

    ax9.plot(t, phi1, label="$\phi_1(t)+t=D_1(t)$", **style2)
    ax9.plot(t, phi2, label="$\phi_2(t)+t=D_2(t)$", **style3)
    ax9.legend(loc="upper right")
    ax9.set_xlabel("t")

    # Add a horizontal legend below all subplots
    fig.legend(handles=[l1, l2, l3],loc='lower center',ncol=3,fontsize=10, frameon=True,fancybox=True, shadow=False,bbox_to_anchor=(0.5, 0.02))
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    plt.show()
 


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

    # Calculate all absolute error data
    error_x = abs(predictor[0:-nD+2, -1, 0] - u_delay[nD-2:, 0])
    error_x_ml = abs(predictor_ml[0:-nD+2, -1, 0] - u_delay[nD-2:, 0])

    error_y = abs(predictor[0:-nD+2, -1, 1] - u_delay[nD-2:, 1])
    error_y_ml = abs(predictor_ml[0:-nD+2, -1, 1] - u_delay[nD-2:, 1])

    error_theta = abs(predictor[0:-nD+2, -1, 2] - u_delay[nD-2:, 2])
    error_theta_ml = abs(predictor_ml[0:-nD+2, -1, 2] - u_delay[nD-2:, 2])

    # Combine all errors to find global min and max
    all_errors = np.concatenate([error_x, error_x_ml, error_y, error_y_ml, error_theta, error_theta_ml])
    ymin, ymax = np.min(all_errors), np.max(all_errors)

    ax6.plot(t[0:-nD+2], abs(predictor[0:-nD+2, -1, 0]-u_delay[nD-2:, 0]), label="Const delay", **style1)
    ax6.plot(t[0:-nD+2], abs(predictor_ml[0:-nD+2,-1, 0]-u_delay[nD-2:, 0]), **style2)
    ax6.set_ylabel(r"$|x(t+D)-\hat{P}(t)|$")
    ax6.set_xlabel("time")
    ax6.set_xticks([0, 2.5, 5, 7.5, 10])
    ax6.set_ylim(ymin, ymax)

    ax7.plot(t[0:-nD+2], abs(predictor[0:-nD+2, -1, 1]-u_delay[nD-2:, 1]), label="Const delay", **style1)
    ax7.plot(t[0:-nD+2], abs(predictor_ml[0:-nD+2,-1, 1]-u_delay[nD-2:, 1]), **style2)
    ax7.set_ylabel(r"$|y(t+D)-\hat{P}(t)|$")
    ax7.set_xlabel("time")
    ax7.set_xticks([0, 2.5, 5, 7.5, 10])
    ax7.set_ylim(ymin, ymax)

    ax8.plot(t[0:-nD+2], abs(predictor[0:-nD+2, -1, 2]-u_delay[nD-2:, 2]), label="Const delay", **style1)
    ax8.plot(t[0:-nD+2], abs(predictor_ml[0:-nD+2,-1, 2]-u_delay[nD-2:, 2]), **style2)
    ax8.set_xlabel("time")
    ax8.set_ylabel(r"$|\theta(t+D)-\hat{P}(t)|$")
    ax8.set_xticks([0, 2.5, 5, 7.5, 10])
    ax8.set_ylim(ymin, ymax)
    # Add a horizontal legend below all subplots
    fig.legend(handles=[l1, l2],loc='lower center',ncol=3,fontsize=10, frameon=True,fancybox=True, shadow=False,bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    plt.show()

def count_parameters(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
