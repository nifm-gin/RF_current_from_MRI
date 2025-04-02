import numpy as np
from colorsys import hls_to_rgb
from utils.utils import ProcessingOptions
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import matplotlib.cm as cm


def scale_data(data, processing_options):
    """
    Construct a Normalization function to fit data to the display color range
    Args:
        data: Data to normalize
        processing_options: Class with members plot_mode and color_map.
            plot_mode: Either of "M" for magnitude, "R" for real, "I" for imaginary or "C" for complex.
                       When specifying "C" (or anything else), real and imaginary parts are scaled by the same value.

    Returns:
        data: Component of the data to be plotted (np.abs(data) for "M", data.real for "R" and data.imag for "I")
        norm: A matplotlib.colors.Normalize object that can be used to plot correct color bars
    """
    plot_mode = processing_options.plot_mode
    v_max = 5
    # Choose which normalizer to use and extract appropriate part of the signal
    if plot_mode == "M":
        norm = cols.Normalize()  # Normalize to linearly map [data.min(), data.max()] to [0, 1]
        data = np.abs(data)
        data[np.isnan(data)] = 0
        norm.autoscale(data)
    elif plot_mode == 'DM':
        # data = np.abs(data)
        # data = abs((data[:, :, :, 0]-data[:, :, :, 1])/data[:, :, :, 0])
        data = data[:, :, :, :, 0] - data[:, :, :, :, 1]
        data = data.reshape(np.shape(data)+(1,))
        v_max = max([np.max(data), np.abs(np.min(data))])
        norm = cols.Normalize(vmin=-v_max, vmax=v_max)  # Normalize to linearly map [data.min(), data.max()] to [0, 1] symetric version
        data[np.isnan(data)] = 0
        # norm.autoscale(data)
    elif plot_mode == "R":
        norm = cols.CenteredNorm(vcenter=0)  # Normalize to map data such that data==0 is mapped to 0.5
        data = data.real
        data[np.isnan(data)] = 0
        norm.autoscale(data)
    elif plot_mode == "DR":
        norm = cols.Normalize()  # Normalize to map data such that data==0 is mapped to 0.5
        data = data.real
        # data = abs((data[:, :, :, 0] - data[:, :, :, 1]) / data[:, :, :, 0])
        data = abs((data[:, :, :, :, 0] - data[:, :, :, :, 1]))
        data = data.reshape(np.shape(data)+(1,))
        data[np.isnan(data)] = 0
        data[data > v_max] = v_max
        data[np.isnan(data)] = 0
        norm.autoscale(data)
    elif plot_mode == "I":
        norm = cols.CenteredNorm(vcenter=0)
        data = data.imag
        data[np.isnan(data)] = 0
        norm.autoscale(data)
    elif plot_mode == "DI":
        norm = cols.Normalize(vmin=0, vmax=5, clip = True)
        data = data.imag
        # data = abs((data[:, :, :, 0] - data[:, :, :, 1]) / data[:, :, :, 0])
        data = abs((data[:, :, :, :, 0] - data[:, :, :, :, 1]))
        data = data.reshape(np.shape(data)+(1,))
        data[np.isnan(data)] = 0
        data[data > v_max] = v_max
        norm.autoscale(data)
    elif plot_mode == "P":
        norm = cols.CenteredNorm(vcenter=0)
        data = np.angle(data)
        data[np.isnan(data)] = 0
        norm.autoscale(data)
    elif plot_mode == "DP":
        norm = cols.Normalize(vmin=0, vmax=5, clip = True)
        data = np.angle(data)
        # data = abs((data[:, :, :, 0] - data[:, :, :, 1]) / data[:, :, :, 0])
        data = abs((data[:, :, :, :, 0] - data[:, :, :, :, 1]))
        data = data.reshape(np.shape(data)+(1,))
        data[np.isnan(data)] = 0
        data[data > v_max] = v_max
        norm.autoscale(data)
    else:
        # Normalize to linearly map the range +-np.max(np.abs(data)) to [-1, 1]
        norm = cols.CenteredNorm(vcenter=0, halfrange=np.max(np.abs(data)))

    return data, norm


def colorize(z, norm):
    """
    Convert complex patch values to RGB triplets for plotting
    Args:
        z: Data to convert
        norm: A normalization function to scale the range of real and imaginary values to [0, 1]

    Returns: Numpy nd-array of size n,m,3, where nxm is the size of the input patch
    """
    # plot complex data as per code from here: https://stackoverflow.com/a/20958684

    # Make array complex if it is not already
    if not np.any(np.iscomplex(z)):
        z = z + 1j * 0
    # scale real and imaginary parts
    z = z / norm.halfrange
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 - 0.5 / (0.5 + r)
    s = 1.0
    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose((1, 2, 0))

    return c


def plot_patch(signal, image_size, n_slices, processing_options, save_fname=None):
    """Plot all slices of acquired or fitted MRI signal or lambda map"""
    signal = signal.T  # shift slice dimension to be the highest dimension
    n_sig_per_seq = processing_options.n_sig_per_seq
    plot_modes = processing_options.plot_mode
    if isinstance(plot_modes, tuple):  # a tuple of plot modes specified
        pm_list = plot_modes
    else:
        pm_list = (plot_modes,)

    if processing_options.model_fit == "C":
        signal = signal.reshape((image_size[0], image_size[1], -1, 2, n_slices))
        signal = signal[:, :, :, 0, :] + 1j * signal[:, :, :, 1, :]
        # signal = signal[:, :, :, 0, :] * np.exp(1j * signal[:, :, :, 1, :])
    else:
        signal = signal.reshape((image_size[0], image_size[1], -1, n_slices))

    # Separate signals from different sequences for display purposes
    n_signals = signal.shape[2]
    if n_signals == 1:
        n_sig_per_seq = 1
    signal = signal.reshape((image_size[0], image_size[1], n_signals // n_sig_per_seq, n_sig_per_seq, n_slices))
    n_sequences = signal.shape[2]
    n_rows = int(np.ceil(np.sqrt(n_slices)))
    n_cols = int(np.ceil(float(n_slices) / n_rows))

    figures = list()
    for this_plot_mode in pm_list:
        if not processing_options.model_fit == "C" and this_plot_mode in ["C", "I", "P", "DP", "DI"]:
            continue
        plot_mode_options = ProcessingOptions(plot_mode=this_plot_mode)

        for i_sequence in range(n_sequences):  # Loop over sequences. Adjust display scale per sequence
            # Make sure we use the same colormap scaling for all slices to be able to compare
            this_signal = signal[:, :, i_sequence, :, :]
            this_signal, norm = scale_data(this_signal, plot_mode_options)
            if this_plot_mode == "P":
                cmap = cm.get_cmap("hsv")
            else:
                cmap = cm.get_cmap(processing_options.plot_color_map)

            sm = cm.ScalarMappable(norm=norm, cmap=cmap)

            for i_signal in range(n_sig_per_seq):  # loop over signals - create one figure per signal
                fig, axes = plt.subplots(n_rows, n_cols)
                for i_slice in range(n_slices):  # loop over slices - create one axis per slice
                    ax = axes[i_slice % n_rows, i_slice // n_rows]
                    if this_plot_mode == "C":
                        ax.imshow(colorize(this_signal[:, :, i_signal, i_slice, ].T, norm), origin="lower", )
                    else:
                        ax.imshow(this_signal[:, :, i_signal, i_slice, ].T, cmap=cmap, norm=norm, origin="lower", )
                    ax.axis('off')
                    ax.set_title("Slice %d" % (i_slice + 1))
                if processing_options.plot_color_bar and not this_plot_mode == "C":
                    plt.colorbar(mappable=sm, ax=axes, orientation="vertical")
                if save_fname is not None:
                    fig.set_size_inches([6, 6])
                    plt.savefig(save_fname + "_%s_seq-%d_sig-%d.png" % (this_plot_mode, i_sequence, i_signal,))
                    plt.clf()
                    plt.close(fig)
                else:
                    figures.append(fig)

    return figures


def plot_compare(signal_1, signal_2, image_size, n_slices, processing_options, save_fname=None, all_slices=False):
    """ Plot for one slice all signals acquired VS fitted"""
    signal_1[np.isnan(signal_2)] = np.nan
    if all_slices:
        slice_to_display = [k for k in range(n_slices)]
    else:
        slice_to_display = [4]
        if n_slices <= slice_to_display:
            slice_to_display = [0]
    n_sig_per_seq = processing_options.n_sig_per_seq
    # n_sig_per_seq = 1
    n_inputs = 2
    n_plot_per_mode=dict()
    n_plot_per_mode["R"] = 2
    n_plot_per_mode["I"] = 2
    n_plot_per_mode["C"] = 2
    n_plot_per_mode["M"] = 2
    n_plot_per_mode["P"] = 2
    n_plot_per_mode["DR"] = 1
    n_plot_per_mode["DI"] = 1
    n_plot_per_mode["DC"] = 1
    n_plot_per_mode["DM"] = 1
    n_plot_per_mode["DP"] = 1
    # n_inputs = 3
    for this_slice in slice_to_display:
        # Keep only selected slice
        this_signal_1 = signal_1[this_slice, :, np.newaxis]
        this_signal_2 = signal_2[this_slice, :, np.newaxis]
        signal = np.concatenate((this_signal_1, this_signal_2), axis=1)
        # signal = np.concatenate((signal_1, signal_2, abs(signal_1-signal_2)/signal_1), axis=1)
        if processing_options.model_fit == "C":
            signal = signal.reshape((image_size[0], image_size[1], -1, 2, n_inputs))
            signal = signal[:, :, :, 0, :] + 1j * signal[:, :, :, 1, :]
            # signal = signal[:, :, :, 0, :] * np.exp(1j * signal[:, :, :, 1, :])
        else:
            signal = signal.reshape((image_size[0], image_size[1], -1, n_inputs))

        # Separate signals from different sequences for display purposes
        n_signals = signal.shape[2]
        signal = signal.reshape((image_size[0], image_size[1], n_signals // n_sig_per_seq, n_sig_per_seq, n_inputs))
        n_sequences = signal.shape[2]

        plot_modes = processing_options.plot_mode
        if isinstance(plot_modes, tuple):  # a tuple of plot modes specified
            pm_list = plot_modes
        else:
            pm_list = (plot_modes,)
        pm_list = pm_list+('DR', 'DI', 'DM', 'DP')

        figures = list()
        for this_plot_mode in pm_list:
            if not processing_options.model_fit == "C" and this_plot_mode in ["C", "I", "P", "DI", "DC", "DP"]:
                continue
            plot_mode_options = ProcessingOptions(plot_mode=this_plot_mode)

            # fig, axes = plt.subplots(n_plot_per_mode[this_plot_mode], n_signals)
            fig, axes = plt.subplots(n_plot_per_mode[this_plot_mode], n_signals)
            # Make sure we use the same colormap scaling for all signals to be able to compare
            normalized_signal, norm = scale_data(signal, plot_mode_options)
            if this_plot_mode == "P":
                cmap = cm.get_cmap("hsv")
            else:
                cmap = cm.get_cmap(processing_options.plot_color_map)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            for i_sequence in range(n_sequences):  # Loop over sequences. Adjust display scale per sequence
                seq_signal = normalized_signal[:, :, i_sequence, :, :]
                for i_signal in range(n_sig_per_seq):
                    for i_input in range(n_plot_per_mode[this_plot_mode]):
                        if n_plot_per_mode[this_plot_mode] == 2:
                            ax = axes[i_input, i_signal + i_sequence * n_sig_per_seq]
                        elif n_plot_per_mode[this_plot_mode] == 1:
                            ax = axes[i_signal + i_sequence * n_sig_per_seq]
                        this_signal = seq_signal[:, :, i_signal, i_input, ].T
                        if this_plot_mode == "C":
                            ax.imshow(colorize(this_signal, norm), origin="lower")
                        else:
                            ax.imshow(this_signal, cmap=cmap, norm=norm, origin="lower")
                        ax.axis('off')
                        # ax.set_title("Sequence %d, Signal %d" % (i_sequence + 1, i_signal + 1,))
            if processing_options.plot_color_bar and this_plot_mode not in "C":
                if n_plot_per_mode[this_plot_mode] == 2:
                    plt.colorbar(mappable=sm,
                                 ax=axes[:, np.arange(n_signals)],
                                 orientation="vertical")
                elif n_plot_per_mode[this_plot_mode] == 1:
                    # im_height = axes[0, 0].get_position().height
                    plt.colorbar(mappable=sm,
                                 ax=axes[np.arange(n_signals)],
                                 orientation="vertical")
                    # cbar.ax.set_aspect(im_height)

            if save_fname is not None:
                fig.set_size_inches([13, 5.25])
                plt.savefig(save_fname + "_%s_%d.png" % (this_plot_mode, this_slice))
                plt.clf()
                plt.close(fig)
            else:
                figures.append(fig)

    return figures


def plot_compare_interface(signal_1, signal_2, image_size, processing_options, fig):
    """ Same as plot_compare but for the interface"""

    slice_to_display = 0
    n_sig_per_seq = processing_options.n_sig_per_seq
    # n_sig_per_seq = 1
    n_inputs = 2
    n_plot_per_mode = 2
    # Keep only selected slice
    signal_1 = signal_1[slice_to_display, :, np.newaxis]
    signal_2 = signal_2[slice_to_display, :, np.newaxis]
    signal = np.concatenate((signal_1, signal_2), axis=1)
    # signal = np.concatenate((signal_1, signal_2, abs(signal_1-signal_2)/signal_1), axis=1)
    if processing_options.model_fit == "C":
        signal = signal.reshape((image_size[0], image_size[1], -1, 2, n_inputs))
        signal = signal[:, :, :, 0, :] + 1j * signal[:, :, :, 1, :]
    else:
        signal = signal.reshape((image_size[0], image_size[1], -1, n_inputs))

    # Separate signals from different sequences for display purposes
    n_signals = signal.shape[2]
    signal = signal.reshape((image_size[0], image_size[1], n_signals // n_sig_per_seq, n_sig_per_seq, n_inputs))
    n_sequences = signal.shape[2]

    plot_mode = processing_options.plot_mode[0]
    if plot_mode == "C":
        plot_mode == "R"

    plot_mode_options = ProcessingOptions(plot_mode=plot_mode)
    fig, axes = plt.subplots(n_plot_per_mode, n_signals)
    fig.set_figwidth(11)
    fig.set_figheight(6)
    for i_sequence in range(n_sequences):  # Loop over sequences. Adjust display scale per sequence
        seq_signal = signal[:, :, i_sequence, :, :]
        seq_signal, norm = scale_data(seq_signal, plot_mode_options)
        if plot_mode == "P":
            cmap = cm.get_cmap("hsv")
        else:
            cmap = cm.get_cmap(processing_options.plot_color_map)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i_signal in range(n_sig_per_seq):
            for i_input in range(n_plot_per_mode):
                if n_plot_per_mode == 2:
                    ax = axes[i_input, i_signal + i_sequence * n_sig_per_seq]
                elif n_plot_per_mode == 1:
                    ax = axes[i_signal + i_sequence * n_sig_per_seq]
                this_signal = seq_signal[:, :, i_signal, i_input, ].T
                if plot_mode == "C":
                    ax.imshow(colorize(this_signal, norm), origin="lower")
                else:
                    ax.imshow(this_signal, cmap=cmap, norm=norm, origin="lower", )
                ax.axis('off')
                ax.set_title("Sequence %d, Signal %d" % (i_sequence + 1, i_signal + 1,))
        if processing_options.plot_color_bar and plot_mode not in "C":
            if n_plot_per_mode == 2:
                plt.colorbar(mappable=sm,
                             ax=axes[:, np.arange(n_sig_per_seq) + i_sequence * n_sig_per_seq],
                             orientation="vertical")
            elif n_plot_per_mode == 1:
                plt.colorbar(mappable=sm,
                             ax=axes[np.arange(n_sig_per_seq) + i_sequence * n_sig_per_seq],
                             orientation="vertical")

    return fig
