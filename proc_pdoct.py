from pathlib import Path

import gc

from numpy_cupy_importer import *
from numba import jit, cuda


from . import calculation
from . import file_tools


class Pdoct(object):
    progress_index = 0

    # function optimized to run on gpu
    # @jit(target_backend ="cuda")
    def process_reference_frame(self, unp, xml, lut, padSize, dispersion_max_order, dispersion_coefficient_range, depth_start_dispersion, depth_end_dispersion, depth_start_saving, depth_end_saving, calibration_signal_offset, reference_frame_index):
        from time import time
        loadparams = time()
        parameters = file_tools.get_parameters(xml)

        num_points = parameters[0]*2
        num_ascans = int(parameters[1]/2)
        num_bscans = parameters[2]

        frame_size = num_points * num_ascans
        offset = frame_size * (reference_frame_index - 1) * cp.dtype(cp.uint16).itemsize

        fid = open(unp, mode='r')
        fid.seek(offset)
        ref_RawData = cp.fromfile(fid, cp.uint16, frame_size).reshape((num_ascans, num_points))
        #ref_RawData = calculation.hilbert_cp(cp.concatenate((, ref_RawData[:,1::2]), axis=1))
        ref_RawData_A = calculation.hilbert_cp(ref_RawData[:,::2])
        ref_RawData_B = calculation.hilbert_cp(ref_RawData[:,1::2])
        loadparams = time() - loadparams

        resamp = time()

        ref_FFTData_A = cp.fft.fft(ref_RawData_A)

        ref_FFTData_B = cp.fft.fft(ref_RawData_B)
        #matlab code does not take in the fft
        rescale_parameter = cp.fromfile(lut, np.double) + 1
        ref_RawData_rescaled_A = calculation.reSampling_LUT(ref_FFTData_A, rescale_parameter, padSize)

        # ref_RawData_rescaled_A = ref_RawData_rescaled_A.T
        ref_RawData_rescaled_B = calculation.reSampling_LUT(ref_FFTData_B, rescale_parameter, padSize)
        # ref_RawData_rescaled_B = ref_RawData_rescaled_B.T

        resamp = time() - resamp

        fpn = time()
        ref_RawData_FPNSub_A = ref_RawData_rescaled_A - (
            cp.tile(calculation.median_cp(cp.real(ref_RawData_rescaled_A), axis=0), [num_ascans, 1]) +
            cp.tile(calculation.median_cp(cp.imag(ref_RawData_rescaled_A), axis=0), [num_ascans, 1]) * 1j
        )
        ref_RawData_FPNSub_B = ref_RawData_rescaled_B - (
            cp.tile(calculation.median_cp(cp.real(ref_RawData_rescaled_B), axis=0), [num_ascans, 1]) +
            cp.tile(calculation.median_cp(cp.imag(ref_RawData_rescaled_B), axis=0), [num_ascans, 1]) * 1j
        )
        ref_RawData_FPNSub = np.concatenate((ref_RawData_FPNSub_A, ref_RawData_FPNSub_B), axis=0)
        fpn = time() - fpn

        hamwin = time()
        window = cp.tile(cp.hanning(int(num_points*0.5)), [num_ascans*2, 1])
        ref_RawData_HamWin = cp.multiply(window, ref_RawData_FPNSub)
        hamwin = time() - hamwin

        disest = time()
        dispersion_coefficients = calculation.get_dispersion_coefficients(ref_RawData_HamWin, dispersion_max_order, dispersion_coefficient_range, depth_start_dispersion, depth_end_dispersion)
        disest = time() - disest

        compdis = time()
        kLinear = cp.linspace(-1, 1, int(num_points*0.5))
        kaxis = cp.tile(kLinear, [num_ascans*2, 1])
        ref_RawData_DisComp = calculation.compensate_dispersion_phase(ref_RawData_HamWin, kaxis, dispersion_max_order, dispersion_coefficients)
        ref_FFTData_DisComp = cp.fft.fft(ref_RawData_DisComp)
        raw_frame = ref_FFTData_DisComp[:, depth_start_saving:depth_end_saving]
        compdis = time() - compdis

        print("loadparams:", loadparams)
        print("resamp:", resamp)
        print("fpn:", fpn)
        print("hamwin", hamwin)
        print("disest", disest)
        print("compdis", compdis)
        return dispersion_coefficients, raw_frame, np.abs(cp.asnumpy(ref_FFTData_DisComp))

    # function optimized to run on gpu
    # @jit(target_backend ="cuda")
    def process_whole_volume(self, unp, xml, lut, dispersion_coefficients, padSize, dispersion_max_order, depth_start_saving, depth_end_saving, calibration_signal_offset, reference_frame_index):
        fid = open(unp, mode='r')
        parameters = file_tools.get_parameters(xml)

        num_points = parameters[0]*2
        num_ascans = int(parameters[1]/2)
        num_bscans = parameters[2]
        num_cscans = parameters[3]
        num_mscans = parameters[4]

        frame_size = num_points * num_ascans
        memory_size = frame_size * num_bscans

        file_pathA = Path(Path(unp).parent / 'CplxProcdData' / Path(unp).stem / (Path(unp).stem + 'A.npy'))
        file_pathB = Path(Path(unp).parent / 'CplxProcdData' / Path(unp).stem / (Path(unp).stem + 'B.npy'))
        ProcdDataA = file_tools.open_memmap(file_pathA, mode='w+', dtype=np.complex64, shape=(num_bscans, num_ascans, depth_end_saving - depth_start_saving))
        ProcdDataB = file_tools.open_memmap(file_pathB, mode='w+', dtype=np.complex64, shape=(num_bscans, num_ascans, depth_end_saving - depth_start_saving))

        for i in range(num_bscans):#range(bscan_size):
            print(i)
            ref_RawData = cp.fromfile(fid, cp.uint16, frame_size).reshape((1, num_ascans, num_points))
            #ref_RawData = calculation.hilbert_cp(cp.concatenate((, ref_RawData[:,1::2]), axis=1))
            ref_RawData_A = calculation.hilbert_cp(ref_RawData[:, :, ::2])
            ref_RawData_B = calculation.hilbert_cp(ref_RawData[:, :, 1::2])
            ref_FFTData_A = cp.fft.fft(ref_RawData_A)

            ref_FFTData_B = cp.fft.fft(ref_RawData_B)
            #matlab code does not take in the fft

            rescale_parameter = cp.fromfile(lut, np.double) + 1
            ref_RawData_rescaled_A = calculation.reSampling_LUT_3d(ref_FFTData_A, rescale_parameter, padSize)
            ref_RawData_rescaled_B = calculation.reSampling_LUT_3d(ref_FFTData_B, rescale_parameter, padSize)

            ref_RawData_FPNSub_A = ref_RawData_rescaled_A - (cp.tile(calculation.median_cp(cp.real(ref_RawData_rescaled_A), axis=1).reshape(1, 1, num_points//2), [1, num_ascans, 1]) + 1j*cp.tile(calculation.median_cp(cp.imag(ref_RawData_rescaled_A), axis=1).reshape(1, 1, num_points//2), [1, num_ascans, 1]))

            ref_RawData_FPNSub_B = ref_RawData_rescaled_B - (cp.tile(calculation.median_cp(cp.real(ref_RawData_rescaled_B), axis=1).reshape(1, 1, num_points//2), [1, num_ascans, 1]) + 1j*cp.tile(calculation.median_cp(cp.imag(ref_RawData_rescaled_B), axis=1).reshape(1, 1, num_points//2), [1, num_ascans, 1]))
            ref_RawData_FPNSub = np.concatenate((ref_RawData_FPNSub_A, ref_RawData_FPNSub_B), axis=1)
            # print("fpn", ref_RawData_FPNSub.shape)
            window = cp.tile(cp.hanning(num_points//2), [1, num_ascans*2, 1])
            ref_RawData_HamWin = cp.multiply(window, ref_RawData_FPNSub)

            kLinear = cp.linspace(-1, 1, num_points//2)
            kaxis = cp.tile(kLinear, [1, num_ascans*2, 1])
            # print("kaxis", kaxis.shape)
            ref_RawData_DisComp = calculation.compensate_dispersion_phase_3d(ref_RawData_HamWin, kaxis, dispersion_max_order, dispersion_coefficients)
            del kLinear
            del kaxis
            gc.collect()
            ref_FFTData_DisComp = cp.fft.fft(ref_RawData_DisComp)
            del ref_RawData_DisComp
            gc.collect()
            ProcdData = cp.asnumpy(ref_FFTData_DisComp[:, :, depth_start_saving:depth_end_saving])
            del ref_FFTData_DisComp
            gc.collect()
            _, end, _ = ProcdData.shape
            ProcdDataA[i,:,:] = ProcdData[:,0:end//2,:]
            ProcdDataB[i,:,:] = ProcdData[:,end//2:end,:]
            del ProcdData
            del end
            gc.collect()

        del ProcdDataA
        del ProcdDataB
        gc.collect()

        from time import time
        save_1 = time()
        ProcdDataA = file_tools.open_memmap(file_pathA, mode='r', dtype=np.complex64, shape=(num_bscans, num_ascans, depth_end_saving-depth_start_saving))
        ProcdDataB = file_tools.open_memmap(file_pathB, mode='r', dtype=np.complex64, shape=(num_bscans, num_ascans, depth_end_saving-depth_start_saving))
        ProcdDataA = calculation.reorder_bscan(ProcdDataA, num_mscans)
        ProcdDataB = calculation.reorder_bscan(ProcdDataB, num_mscans)
        save_1 = time() - save_1
        print("second save: ", save_1)
        return ProcdDataA, ProcdDataB
