import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.colors as mcolors  # 导入正确的 Colormap 类

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'   # 数学公式用 STIX 字体，接近 Times
plt.rcParams['mathtext.rm'] = 'Times New Roman'

def add_colorbar(ax, im, bbox_transform, #ax.transAxes
                 width="5%", 
                 height="100%",
                 loc='lower left',
                 bbox_to_anchor=(1.01, 0., 1, 1),
                 borderpad=0,
                 ctitle=''):
    axins = inset_axes(ax,
                       width=width, 
                       height=height,
                       loc=loc,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=bbox_transform,
                       borderpad=borderpad)
    cbar = plt.colorbar(im, cax=axins)
    axins.set(title=ctitle)
    return cbar

def imagesc(fig,
            images,
            vmin=1.5,
            vmax=4.5,
            extent=[0, 1.01, 1.01, 0],
            aspect=1,
            nRows_nCols=(1, 1),
            cmap='coolwarm',
            ylabel="Depth (km)",
            xlabel="Position (km)",
            fontsize=2,
            xticks=np.arange(0., 1.01, 0.4),
            yticks=np.arange(0., 1.01, 0.4),
            cbar_width="5%",
            cbar_height="100%",
            cbar_loc='lower left',
            cbar_mode="row",  # Updated to row-wise colorbar
            bbox_to_anchor=(1.05, 0., 1, 1.),
            titles=None,
            cbar_titles=None,
            ):
    (nrow, ncol) = nRows_nCols
    
    if not isinstance(vmin, (list, tuple, np.ndarray)):
        vmin = [vmin] * nrow
    if not isinstance(vmax, (list, tuple, np.ndarray)):
        vmax = [vmax] * nrow
    if not isinstance(cmap, (list, tuple, np.ndarray)):
        cmap = [cmap] * nrow
    
    gs = fig.add_gridspec(nrow, ncol)
    for irow in range(nrow):
        for icol in range(ncol):
            ax = fig.add_subplot(gs[irow, icol])
            
            im = ax.imshow(images[irow, icol], 
                           vmin=vmin[irow][icol], vmax=vmax[irow][icol], 
                           extent=extent,
                           aspect=aspect,
                           cmap=cmap[irow])
            
            label_letters = 'abcdefghijklmnopqrstuvwxyz'
            label_idx = irow * ncol + icol
            if label_idx < len(label_letters):
                ax.text(-0.01, 1.15, f'({label_letters[label_idx]})', 
                    transform=ax.transAxes, 
                    fontsize=fontsize, 
                    va='top', ha='left', color='k')
            
            if titles is not None:
                title_idx = irow * ncol + icol
                if title_idx < len(titles):
                    ax.set_title(titles[title_idx], fontsize=fontsize + 1)

            if icol == 0:
                ax.set_ylabel(ylabel, fontsize=fontsize)
                if yticks is not None:
                    ax.set_yticks(yticks)
            else:
                ax.set_yticks([])

            if irow == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                if xticks is not None:
                    ax.set_xticks(xticks)
            else:
                ax.set_xticks([])
            
            ax.tick_params(axis='both', labelsize=fontsize, which='major', pad=0.1)

            # Add colorbar for each row at the last column
            
            axins = inset_axes(ax,
                                width=cbar_width, 
                                height=cbar_height,
                                loc=cbar_loc,
                                bbox_to_anchor=bbox_to_anchor,
                                bbox_transform=ax.transAxes,
                                borderpad=0,
                                )
            axins.tick_params(axis='both', labelsize=fontsize, which='major', pad=0.1)
            cbar = plt.colorbar(im, cax=axins)
            cbar.ax.tick_params(labelsize=fontsize-2),

            if cbar_titles is not None:
                title_idx = irow * ncol + icol
                if title_idx < len(cbar_titles):
                     cbar.ax.set_title(cbar_titles[title_idx],fontsize=fontsize-2,pad=5) 
    
    return

def imagesc1(fig,
            images,
            vmin=1.5,
            vmax=4.5,
            extent=[0, 1.01, 1.01, 0],
            aspect=1,
            nRows_nCols=(1, 1),
            cmap='coolwarm',
            ylabel="Depth (km)",
            xlabel="Position (km)",
            clabel="km/s",
            fontsize=2,
            xticks=np.arange(0., 1.01, 0.4),
            yticks=np.arange(0., 1.01, 0.4),
            cbar_width="5%",
            cbar_height="100%",
            cbar_loc='lower left',
            cbar_mode="row",  # Updated to row-wise colorbar
            bbox_to_anchor=(1.05, 0., 1, 1.),
            titles=None,
            ):
    (nrow, ncol) = nRows_nCols
    
    if not isinstance(vmin, (list, tuple, np.ndarray)):
        vmin = [vmin] * nrow
    if not isinstance(vmax, (list, tuple, np.ndarray)):
        vmax = [vmax] * nrow
    if not isinstance(cmap, (list, tuple, np.ndarray)):
        cmap = [cmap] * nrow
    
    gs = fig.add_gridspec(nrow, ncol)
    for irow in range(nrow):
        for icol in range(ncol):
            ax = fig.add_subplot(gs[irow, icol])
            
            im = ax.imshow(images[irow, icol], 
                           vmin=vmin[irow], vmax=vmax[irow], 
                           extent=extent,
                           aspect=aspect,
                           cmap=cmap[irow])
            
            label_letters = 'abcdefghijklmnopqrstuvwxyz'
            label_idx = irow * ncol + icol
            if label_idx < len(label_letters):
                ax.text(-0.01, 1.09, f'({label_letters[label_idx]})', 
                    transform=ax.transAxes, 
                    fontsize=fontsize+2, 
                    va='top', ha='left', color='k')
            
            if titles is not None:
                title_idx = irow * ncol + icol
                if title_idx < len(titles):
                    ax.set_title(titles[title_idx], fontsize=fontsize + 2)

            if icol == 0:
                ax.set_ylabel(ylabel, fontsize=fontsize)
                if yticks is not None:
                    ax.set_yticks(yticks)
            else:
                ax.set_yticks([])

            if irow == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                if xticks is not None:
                    ax.set_xticks(xticks)
            else:
                ax.set_xticks([])
            
            ax.tick_params(axis='both', labelsize=fontsize, which='major', pad=0.1)

            # Add colorbar for each row at the last column
            if icol == ncol - 1:
                axins = inset_axes(ax,
                                   width=cbar_width, 
                                   height=cbar_height,
                                   loc=cbar_loc,
                                   bbox_to_anchor=bbox_to_anchor,
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )
                axins.tick_params(axis='both', labelsize=fontsize, which='major', pad=0.1)
                cbar = plt.colorbar(im, cax=axins)
                cbar.ax.set_ylabel(clabel, fontsize=fontsize-3)
    
    return






