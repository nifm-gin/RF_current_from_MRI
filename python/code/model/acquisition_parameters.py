import numpy as np


class AcquisitionParameters:
    """Class aimed to store MRI sequence parameters"""

    def __init__(
        self,
        flip1: list = None,
        flip2: list = None,
        tr1: list = None,
        tr2: list = None,
        n: list = None,
        te: float = None,
        te_ste: float = None,
        td: float = None,
        ts: float = None,
    ):
        """Constructor of AcquisitionParameters (designed for AFI, da-hdrAFI, DREAM or SPGR sequences

                Args:
                    flip1 (list): first flip angle (all sequences)
                    flip2 (list): second flip angle (da-hdrAFI)
                    tr1 (list): first repetition time (all sequences)
                    tr2 (list): second repetition time (AFI and da-hdrAFI)
                    n (list): ratio tr2/tr1 (AFI and da-hdrAFI)
                    te (float): echo time (all sequences)
                    te_ste (list): stimulated echo echo time (DREAM)
                    td (float): effective time delay between STEAM preparation and low-angle singleshot imaging sequence (DREAM)
                    ts (float): STEAM preparation seperation time interval (DREAM)
                    """

        if flip1 is None:
            flip1 = []
        self.flip1 = flip1
        if flip2 is None:
            flip2 = []
        self.flip2 = flip2
        if tr1 is None:
            tr1 = []
        self.tr1 = tr1
        if tr2 is None:
            tr2 = []
        self.tr2 = tr2
        if n is None:
            n = []
        self.n = n
        self.te = te
        self.te_ste = te_ste
        self.td = td
        self.ts = ts
        self.n_signals = None
        self.signal_functions = []
