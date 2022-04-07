import processing.proc_pdoct as pdoct
import matplotlib.pyplot as plt

#
from PyQt5 import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore
import sys
from pathlib import Path
import gc
import numpy as np
from os import path
import os
from pathlib import Path
import csv
import cupy as cp
import time
from processing import file_tools
#

unp = "C:\\Users\\coil_\\Desktop\\Eunice\\data\\16_28_53-.unp"
xml = "C:\\Users\\coil_\\Desktop\\Eunice\\data\\16_28_53-.xml"
lut = "C:\\Users\\coil_\\Desktop\\Eunice\\data\\LUTSS.bin"
dispersion_max_order = 4
dispersion_coefficient_range = 10
depth_start_dispersion = 30
depth_end_dispersion = 3001
depth_start_saving = 1
depth_end_saving = 600
calibration_signal_offset = 20
reference_frame_index = 10
padSize = 4096

obj = pdoct.Pdoct()
import sys
sys.stdout = open('STDOUT.txt', 'w+')
dispCoeffs, rawFrame, ref_FFT = obj.process_reference_frame(unp, xml, lut, padSize, dispersion_max_order, dispersion_coefficient_range, depth_start_dispersion, depth_end_dispersion, depth_start_saving, depth_end_saving, calibration_signal_offset, reference_frame_index)


import matplotlib.pyplot as plt
# plt.imshow(20*np.log10(abs(ref_FFT[:, :].T)), cmap='gray')
# print(ref_FFT.shape)
# plt.show()
# plt.show()

ProcdDataA, ProcdDataB = obj.process_whole_volume(unp, xml, lut, dispCoeffs, padSize, dispersion_max_order, depth_start_saving, depth_end_saving, calibration_signal_offset, reference_frame_index)

from tifffile import *

imwrite('pdoctA.tif', np.flip(20*np.log10(abs(ProcdDataA))), dtype=np.uint16)
imwrite('pdoctB.tif', np.flip(20*np.log10(abs(ProcdDataB))), dtype=np.uint16)


sys.stdout.close()
