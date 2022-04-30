import os
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
import xml.etree.ElementTree as ET
from functions import hilbert
from functions import hanning
from functions import resamplingLUT
from functions import setDispCoeff
from functions import compDisPhase
from functions import mat2gray
from functions import imadjust
from functions import reorderBscan
from numpy.fft import fft
import numpy.matlib
import matplotlib.pyplot as plt
from PIL import Image
# import imagesc as imagesc

class OCT():
    """ """
    def __init__(self, path, disp_max_order = 5, coeff_range = 50, cal_sig_off_set_idx = 15):
        self.coeff_range = coeff_range
        self.disp_max_order = disp_max_order
        self.cal_sig_off_set_idx = cal_sig_off_set_idx
        self.unp = Path(path)
        self.xml = os.path.join(self.unp.parent, str(self.unp.stem) +".xml")

    """ """
    def parse_xml(self):
        try:
            self.tree = ET.parse(self.xml)
        except IOError:
            print("File does not exist or something happened.")

        self.root = self.tree.getroot()

        try:
            self.xml_name = self.root[0][0].text
        except IndexError:
            print("Likely, \"Name\" tag not found in XML.")

        try:
            self.xml_time = self.root[0][1].attrib
        except IndexError:
            print("Likely, \"Time\" tag not found in XML.")

        try:
            self.xml_vol_sz = self.root[0][2].attrib
        except IndexError:
            print("Likely, \"Volume Size\" tag not found in XML.")

        try:
            self.xml_scan_params = self.root[0][3].attrib
        except IndexError:
            print("Likely, \"Scanning Parameters\" tag not found in XML.")

        try:
            self.xml_disp_params = self.root[0][4].attrib
        except IndexError:
            print("Likely, \"Dispersion Parameters\" tag not found in XML.")

        try:
            self.xml_fix_target = self.root[0][5].attrib
        except IndexError:
            print("Likely, \"Fixation Target\" tag not found in XML.")

    """ """
    def load_params(self):
        try:
            self.params = {}
            self.params["num_pts"] = int(self.xml_vol_sz["Width"])*2
            self.params["num_a_scans"] = int(self.xml_vol_sz["Height"])/2
            self.params["num_b_scans"] = int(self.xml_vol_sz["Number_of_Frames"])
            self.params["num_c_scans"] =  int(self.xml_vol_sz["Number_of_Volumes"])
            self.params["num_m_scans"] = int(self.xml_scan_params["Number_of_BM_scans"])

        except Exception:
            print("Error. Make sure XML parsed correctly.")

    """ """
    def read_uint8(self, file):
        f = open(file, 'rb')
        self.LUT = np.fromfile(f, np.double) # typecast to double
        print(self.LUT)
        f.close()

    # section 6, 7, 8
    """ """
    def ref_frame(self, frame):
        f = open(self.unp, 'rb')
        f.seek(int(2*self.params["num_pts"]*self.params["num_a_scans"]*(frame-1)),0)
        self.refRaw_interlace = np.fromfile(f, count=int(self.params["num_a_scans"])*int(self.params["num_pts"]), dtype="uint16").reshape((int(self.params["num_a_scans"]), int(self.params["num_pts"])))
        self.refRaw_interlace = np.transpose(self.refRaw_interlace)
        self.refRaw = np.concatenate((self.refRaw_interlace[::2,:], self.refRaw_interlace[1::2,:]), axis=1)
        #plt.plot(np.real(self.refRaw[:,0]))
        self.refRaw_cplx = hilbert(self.refRaw)
        print(self.refRaw_cplx[0,0])
        plt.plot(abs(fft(self.refRaw_cplx[:,0], axis=0)))
        plt.show()
        # hilbert in python != hilbert in MATLAB
        f.close()

    X = np.array([0, 1, 2, 3, 4])
    print(X[::2], X[1::2])
    """ """
    def rescale(self):
        self.refRaw_rescaled = resamplingLUT(self.refRaw, self.LUT)
        self.fft_rescaled = fft(self.refRaw_rescaled)

    """ """
    def removeFPN(self):
        halfN = int(self.refRaw_rescaled.shape[1]/2)
        N = self.refRaw_rescaled.shape[1]
        subA = list(range(0, halfN))
        subB = list(range(halfN, N))

        self.refRaw_FPNSub_A = self.refRaw_rescaled[:, subA] - np.matlib.repmat(np.median(np.real(self.refRaw_rescaled[:,subA]).shape[1]), 1,self.refRaw_rescaled[:,subA].shape[1])
        self.refRaw_FPNSub_A += np.multiply(1j, np.matlib.repmat(np.median(np.imag(self.refRaw_rescaled[:,subA]).shape[1]), 1,self.refRaw_rescaled[:,subA].shape[1]))
        self.refRaw_FPNSub_B = self.refRaw_rescaled[:, subB] - np.matlib.repmat(np.median(np.real(self.refRaw_rescaled[:,subB]).shape[1]), 1,self.refRaw_rescaled[:,subB].shape[1])
        self.refRaw_FPNSub_B += np.multiply(1j, np.matlib.repmat(np.median(np.imag(self.refRaw_rescaled[:,subB]).shape[1]), 1,self.refRaw_rescaled[:,subB].shape[1]))

        self.refRaw_FPNSub = np.concatenate((self.refRaw_FPNSub_A, self.refRaw_FPNSub_B), axis=1)
        #self.fft_FPNSub = fft(self.refRaw_FPNSub)

    def windowing(self):
        win = (np.float64(hanning(self.refRaw_FPNSub.shape[0])))
        win = win.reshape(win.shape[0], 1)
        # print(np.matlib.repmat(win, 1, self.refRaw_FPNSub.shape[1]).shape)
        self.refRaw_HamWin = self.refRaw_FPNSub * np.matlib.repmat(win, 1, self.refRaw_FPNSub.shape[1])
        self.fft_HamWin = fft(self.refRaw_HamWin)

        # NOT WORKING idek why !

        # print(self.refRaw_interlace)
        # toshow = mat2gray(20*np.log10(abs(fft(self.refRaw_interlace))))
        # print(toshow)
        # #toshow2 = imadjust(toshow,  min(toshow), max(toshow), 0, 1)
        # #print(toshow2)
        # fig = plt.figure()
        # fig.suptitle('refraw interlace plot')
        # plt.imshow(20*np.log10(abs(fft(self.refRaw_interlace))), cmap="gray")
        # plt.show()
        # # plt.show()
        # # plt.imshow(20*np.log10(abs(self.fft_HamWin)), cmap='gray')
        # # # plt.imshow(imadjust(mat2gray(20*np.log10(abs(self.fft_HamWin)))), extent=[-1, 1, -1, 1], cmap='gray')
        # # plt.show()

    def dispEstimate(self, dispROI0=150, dispROI1=500):
        self.dispROI = [150, 500]
        self.dispCoeffs_A, self.kaxis = setDispCoeff(self.refRaw_HamWin, self.disp_max_order, self.coeff_range, self.dispROI[0], self.dispROI[1])
        self.ref_RawData_DisComp = compDisPhase(self.refRaw_HamWin, self.kaxis, self.disp_max_order, self.dispCoeffs_A)
        print(self.ref_RawData_DisComp.shape)
        # self.ref_FFT_Final   = fft(self.ref_RawData_DisComp);
        # self.ref_OCT_Log     = 20.*log10(abs(self.ref_FFT_Final))
        # Dispersion estimation & compensation

    def volFrame(self, frame):
        f.seek(int(2*self.params["num_pts"]*self.params["num_a_scans"]*(frame-1)),0)
        self.refRaw_interlace = np.fromfile(f, count=int(self.params["num_a_scans"])*int(self.params["num_pts"]), dtype="uint16").reshape((int(self.params["num_a_scans"]), int(self.params["num_pts"])))
        self.refRaw_interlace = np.transpose(self.refRaw_interlace)
        self.refRaw = hilbert(np.concatenate((self.refRaw_interlace[::2,:], self.refRaw_interlace[1::2,:]), axis=1))

    def volEstimation(self):
        return fft(compDisPhase(self.refRaw_HamWin, self.kaxis, self.disp_max_order, self.dispCoeffs_A))

""" """
def volProc(unp='16_28_53-.unp', unt='LUTSS.bin', ptLim=1100):
    #  extra uneccessary time if a new alg. can be found!
    OCTvol = OCT(unp)
    OCTvol.parse_xml()
    OCTvol.load_params()
    OCTvol.read_uint8(unt)
    # end of extra
    procData = np.zeros((ptLim,
                        int(OCTvol.params['num_a_scans']*2),
                        int(OCTvol.params['num_b_scans'])))
    print(int(OCTvol.params["num_b_scans"]))

    # prelim run
    OCTvol.ref_frame(10+round(OCTvol.params["num_b_scans"]/2)) # OCT RefFrame of FrameNum
    OCTvol.rescale()
    OCTvol.removeFPN()
    OCTvol.windowing()
    OCTvol.dispEstimate()

    for frameNum in range(1, 1+int(OCTvol.params["num_b_scans"])): # problem? 0 to bScan-1
        OCTvol.volFrame(frameNum) # OCT RefFrame of FrameNum
        OCTvol.rescale()
        OCTvol.removeFPN()
        OCTvol.windowing()
        print("reached1.")
        start = time()
        fftData_DispComp = OCTvol.volEstimation()
        spr1 = time() - start
        print("reached2: ", spr1)
        print(procData.shape)
        procData[:,:,frameNum] = fftData_DispComp[0:1100,:]
        print('OCT volume process: %d\n', frameNum);

    return procData, OCTvol.params["num_m_scans"]

""" """
def save(procData, mScans, depthROI=[1, 150], fileA="OCTA.tiff", fileB="OCTB.tiff"):
    m,n,o = procData.shape
    A = list(range(1, int(n/2)))
    B = list(range(int(n/2)+1, n))
    cplxDataA = reorderBscan(procData[depthROI[0]:depthROI[1], A, :], mScans)
    cplxDataB = reorderBscan(procData[depthROI[0]:depthROI[1], B, :], mScans)
    cplxDataA_log  = np.multiply(20, np.log10(np.absolute(cplxDataA)))
    cplxDataB_log  = np.multiply(20, np.log10(np.absolute(cplxDataB)))
    cplxDataA_logFlip = np.flip(cplxDataA_log)
    cplxDataB_logFlip = np.flip(cplxDataB_log)
    # Save OCTA
    OCTA = Image.fromarray(cplxDataA_logFlip)
    OCTA.save('OCTA.tif')
    # Save OCTB
    OCTB = Image.fromarray(cplxDataB_logFlip)
    OCTB.save('OCTB.tif')
    # SAVE AS TIFF
    # [~,fname_save,~] = fileparts(fn);
    # exportTiff(flip(20*log10(abs(cplxDataA))),[fname_save,'OCT_A'])
    # exportTiff(flip(20*log10(abs(cplxDataB))),[fname_save,'OCT_B'])
    # save(fullfile(cd,[fname_save,'A']), 'cplxData_A', '-v7.3');
    # save(fullfile(cd,[fname_save,'B']), 'cplxData_B', '-v7.3');


# start = time()
# OCTA = OCT('16_28_53-.unp')
# OCTA.parse_xml()
# sprint1 = time() - start
#
# start = time()
# OCTA.load_params()
# sprint2 = time() - start
#
# start = time()
# OCTA.read_uint8('LUTSS.bin')
# sprint3 = time() - start
#
# start = time()
# OCTA.ref_frame(10+round(OCTA.params["num_b_scans"]/2)
# sprint3 = time() - start
#
# start = time()
# OCTA.rescale()
# sprint4 = time() - start
#
# start = time()
# OCTA.removeFPN()
# sprint5 = time() - start
#
# start = time()
# OCTA.windowing()
# sprint6 = time() - start
#
# start = time()
# OCTA.dispEstimate()
# sprint7 = time() - start

start = time()
procData, mScans = volProc()
save(procData, mScans)
sprint8 = time() - start

print(sprint8)
#print(sprint1, sprint2, sprint3, sprint4, sprint5, sprint6, sprint7, sprint8)
