# import math
import numpy as np
# import time


# pythran export afi_precalc_1alpha(float32, float32, float32, float32, int64, int64, float32, float32, float32)
def afi_precalc_1alpha(e1, e2, sa, ca, extent):
    """
    Calculate part of the AFI signals for the given slice profile dictionary and pulse sequence parameters. This
    excludes scaling by signal_sign and exp(-TE/T2*).

    Parameters
    ----------
    e1 : np.ndarray
        Precalculated value of exp(-tr1 / t1) for the entire grid. This is a vector of size (n_subgrid, 1,).
    e2 : np.ndarray
        Precalculated value of exp(-tr2 / t1) for the entire grid. This is a vector of size (n_subgrid, 1,).
    sa : np.ndarray
        Real transverse component of the slice profile dictionary, containing the fraction of longitudinal magnetization
        transformed by the excitation RF pulse into real transverse magnetization across the slice (sin(alpha), alpha
        being the local flip angle). This is a matrix of size (1, n_flip, n_z,). n_flip is the number of entries in the
        slice profile dictionary as a function of the actual flip angle at the center of the slice and n_z is the number
        of positions across the slice in the slice profile
    ca : np.ndarray
        Longitudinal component of the slice profile dictionary, containing the fraction of longitudinal magnetization
        left after the excitation RF pulse (cos(alpha), alpha being the local flip angle). Same shape as sa
    extent : float
        Width of the domain represented in the slice profile relative to the nominal slice thickness. This is used to
        scale the signal amplitude obtained by averaging of the entire slice profile, in order to provide the average
        signal amplitude over the thickness of the slice.

    Returns
    -------
    signals : list
        The calculated AFI signals in a list containing signal_1 and signal_2, each being a np.ndarray of size
        (n_grid, n_vox,).
    """
    n_z = sa.shape[-1]

    e1e2 = e1 * e2
    sf = sa * (extent / n_z) / (1 - e1e2 * ca**2)
    signal_1 = sf * ((1 - e2) + (e2 - e1e2) * ca)
    signal_1 = np.einsum(
        "...i->...", signal_1
    )  # sum over last axis (z-positions). Same as np.sum(s, -1), but faster
    signal_2 = sf * ((1 - e1) + (e1 - e1e2) * ca)
    signal_2 = np.einsum(
        "...i->...", signal_2
    )  # sum over last axis (z-positions). Same as np.sum(s, -1), but faster

    return signal_1, signal_2


# pythran export afi_precalc_2alpha(float32, float32, float32, float32, float32, float32, float32)
def afi_precalc_2alpha(e1, e2, sa1, sa2, ca1, ca2, extent):
    """
        Calculate part of the da-hdrAFI signals for the given slice profile dictionary and pulse sequence parameters. This
        excludes scaling by signal_sign and exp(-TE/T2*).

        Parameters
        ----------
        e1 : np.ndarray
            Precalculated value of exp(-tr1 / t1) for the entire grid. This is a vector of size (n_subgrid, 1,).
        e2 : np.ndarray
            Precalculated value of exp(-tr2 / t1) for the entire grid. This is a vector of size (n_subgrid, 1,).
        sa1 : np.ndarray
            Real transverse component of the slice profile dictionary, containing the fraction of longitudinal magnetization
            transformed by the first excitation RF pulse into real transverse magnetization across the slice (sin(alpha1), alpha1
            being the local flip angle). This is a matrix of size (1, n_flip, n_z,). n_flip is the number of entries in the
            slice profile dictionary as a function of the actual flip angle at the center of the slice and n_z is the number
            of positions across the slice in the slice profile
        sa2 : np.ndarray
            Real transverse component of the slice profile dictionary, containing the fraction of longitudinal magnetization
            transformed by the second excitation RF pulse into real transverse magnetization across the slice (sin(alpha2), alpha2
            being the local flip angle). This is a matrix of size (1, n_flip, n_z,). n_flip is the number of entries in the
            slice profile dictionary as a function of the actual flip angle at the center of the slice and n_z is the number
            of positions across the slice in the slice profile
        ca1 : np.ndarray
            Longitudinal component of the slice profile dictionary, containing the fraction of longitudinal magnetization
            left after the first excitation RF pulse (cos(alpha1), alpha1 being the local flip angle). Same shape as sa
        ca2 : np.ndarray
            Longitudinal component of the slice profile dictionary, containing the fraction of longitudinal magnetization
            left after the second excitation RF pulse (cos(alpha2), alpha2 being the local flip angle). Same shape as sa
        extent : float
            Width of the domain represented in the slice profile relative to the nominal slice thickness. This is used to
            scale the signal amplitude obtained by averaging of the entire slice profile, in order to provide the average
            signal amplitude over the thickness of the slice.

        Returns
        -------
        signals : list
            The calculated da-hdrAFI signals in a list containing signal_1 and signal_2, each being a np.ndarray of size
            (n_grid, n_vox,).
        """
    n_z = sa1.shape[-1]

    e1e2 = e1 * e2
    # scale with extent to avoid a bias when calculating the mean below, since only 1/extent is inside the slice
    sf = (extent / n_z) / (1 - e1e2 * (ca1 * ca2))
    signal_1 = sf * sa1 * ((1 - e2) + (e2 - e1e2) * ca2)
    signal_1 = np.einsum(
        "...i->...", signal_1
    )  # sum over last axis (z-positions). Same as np.sum(s, -1), but faster
    signal_2 = sf * sa2 * ((1 - e1) + (e1 - e1e2) * ca1)
    signal_2 = np.einsum("...i->...", signal_2)

    return signal_1, signal_2
