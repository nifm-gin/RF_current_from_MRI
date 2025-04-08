# Code associated with the article *Assessing radiofrequency safety of active implants by measuring induced radiofrequency currents using MRI*
The context of use of this code is in the area of MRI safety for active implants. This code corresponds to the data processing of a proposed method to better individualize MRI exposure for wire-like implant carriers. For more details, see the article (not published yet).
This repository contains 2 parts. The first one is dedicated to the RF current fitting from MRI signals. The second is used to build the dictionary of MRI signals needed to fit the current from signal acquired with given sequence-dependent parameters.

## RF current fitting from MRI signals
### Installation
#### Get sources from git

Clone the rf_current_from_mri repository:
```shell script
git clone https://github.com/nifm-gin/RF_current_from_MRI.git
```

You will need to add the `code` subdirectory to the python path below. Make note of that directory. To find the exact directory to add, you can run
```shell script
echo `pwd`/RF_current_from_MRI/python/code
```
in the same directory where you performed the `git clone` command above.

Define the environment variable RF_CURRENT_FROM_MRI_DATA to point to the data directory for this project.
To find the directory with provided test data and set the environnement variable:
```
echo `pwd`RF_current_from_MRI/python/data
export RF_CURRENT_FROM_MRI_DATA="/path/to/data"
```
Alternatively, you can create a symbolic link in your home, called "rf_current_from_mri_data" and pointing to the desired directory.

#### Basic installation for Python 3.11

Using Python 3.11 or later is recommended if available, since it runs much faster than
earlier versions of Python3.

Create a new virtual environment for python using Miniconda or virtualenvwrapper.
These instructions are for virtualenvwrapper. If necessary, get the virtualenvwrapper script from the system installation directory
```shell script
cp /usr/share/virtualenvwrapper/virtualenvwrapper.sh /home/username/.local/bin/virtualenvwrapper.sh
```

Source the virtualenvwrapper configuration scripts. It is useful to add this line to the end of 
the ~/.bashrc
```shell script
source ~/.local/bin/virtualenvwrapper.sh
```
Create virtual environment for this project. For example if using virtualenv:
```shell script
mkvirtualenv -p /usr/bin/python3.11 rf_current_from_mri_311
```

Activate the virtual environment for this project if not already active.
```shell script
workon rf_current_from_mri_311
```

Add the `code` directory to the python path of the virtual environment by running
```shell script
add2virtualenv /full/path/to/code
```

Install required packages
```shell script
pip3 install --upgrade pip
pip3 install math numpy h5py matplotlib regex sobol_seq nibabel json os scipy gc subprocess tkinter time abc colorsys
```

### Application
All files related to this part are in the "python" directory. The "dico_bloch/signal_dict" directory contains already calculated MRI signals dictionaries for the da-hdrAFI described in the article. The "data" directory contains data to test the method, corresponding to the one showed in the article. The "code" directory contains all python file to apply the RF current measurement method.
The main file is "current_from_acquired_data". In order to launch a current fitting from test data, at the end of the file the call for the selected configuration (copper wire/DBS lead/simulated copper wire and its position as described in the article) has to be uncommented before running this file. At the end of the execution, results are stored in "results/test_object" directory. These results contains a .txt file with both input and fitted parameters ; plots that compare acquired and fitted signals for each slice ; errors between acquired and fitted signals.
Example of code to uncomment to run the analysis of the 5th position of the DBS lead:
```
# DBS lead position  5
current_from_acquired_data(
    [data_dir + '/DBS_lead/23-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',
     data_dir + '/DBS_lead/24-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],
    [113.1, 67.8, 0.5, 114.2, 66.9, 11.9])
```
Arguments of this function are the path to the data of both part of the da-hdrAFI acquisition and the coordinates in voxels of the center of the wire.

## Building da-hdrAFI signal dictionaries
### Installation
A matlab license is needed to use this code.
### Application
The main file that calculates the da-hdrAFI taking into account slice profile effect due to the RF pulse is "build_dictionary_dahdrAFI". In order to obtain a dictionary that can be used in "current_from_acquired_data", this file should be launched with the target parameters (sequence-dependent parameters as described in the article, relaxation time T1 and T2*, repartition of spins parameters and RF pulse parameters). The resulting dictionaries are stored in the working directory and should be moved to "python/dico_bloch" to be used. 

