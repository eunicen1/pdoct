import numpy as np
from numpy import *
from numpy.fft import fft
from numpy.fft import ifft
from scipy.interpolate import interp1d
import cv2

def hilbert(x):
    N = x.shape[-1]
    Xf = fft(x)
    h = np.zeros(N)

    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[-1] = slice(None)
        h = h[tuple(ind)]

    x = ifft(np.multiply(Xf, h))

    return x

def hanning(M):
    if M < 1:
        return array([], dtype=np.result_type(M, 0.0))
    if M == 1:
        return ones(1, dtype=np.result_type(M, 0.0))
    n = np.arange(1-M, M, 2)
    ret = 0.5
    inner = np.multiply(np.pi, n)
    inner = np.divide(inner, (M-1))
    ret = np.multiply(ret, 1 + np.cos(inner))
    return ret
    #return 0.5 + 0.5*cos(pi*n/(M-1))

def mat2gray(A):
    # alpha = min(A.flatten())
    # beta = max(A.flatten())
    # I = A
    # cv2.normalize(A, I, alpha, beta ,cv2.NORM_MINMAX)
    # I = np.uint8(I)
    return (1-A)*255

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    print(y)
    return y

""" """
def resamplingLUT(raw, rescaleParam):
    fftD = fft(raw)

    fftD_r = np.pad(np.real(fftD), ((0,(np.power(2,15) - fftD.shape[0])), (0,0)), 'constant')
    fftD_i = np.pad(np.imag(fftD), ((0,(np.power(2,15) - fftD.shape[0])), (0,0)), 'constant')
    fftD_pad = fftD_r + np.multiply(1j, fftD_i)

    rawD_r = np.real(ifft(fftD_pad))
    rawD_i = np.imag(ifft(fftD_pad))
    rawD_rescaled = np.zeros([len(rescaleParam), fftD.shape[1]], dtype=complex)
    # skipped idxPixel cause never used

    for idxD in range(0, fftD.shape[1]):
        rp = rescaleParam
        rp = np.multiply(rp, fftD_pad.shape[0])
        rp = np.divide(rp, fftD.shape[0])
        spanr0 = np.array(list(range(rawD_r.shape[0])))
        spani0 = np.array(list(range(rawD_i.shape[0])))
        rep = interp1d(np.transpose(spanr0),rawD_r[:,idxD],kind='cubic')(rp)
        imp = np.multiply(1j, interp1d(np.transpose(spani0),rawD_i[:,idxD],kind='cubic')(rp))

        rawD_rescaled[:,idxD] = rep + imp
        # interp1d CUBIC may give different results
    return rawD_rescaled

""" """
def setDispCoeff(inputData, maxDispOrders, coeffRange, start_depth, end_depth):
    frame_data = inputData
    arrCountDispCoeff = np.zeros(maxDispOrders - 1)
    [sizeC, sizeR] = inputData.shape
    kLinear = np.linspace(-1, 1, sizeR)
    kaxis = np.tile(kLinear, [sizeC, 1])
    for i in range(maxDispOrders - 1):
        arrCost = np.zeros(50)
        arrCost_append = np.zeros(1)
        arrDispCoeffRng = np.linspace(-1 * coeffRange, coeffRange, 50)
        arrDispCoeffRng = np.pad(arrDispCoeffRng, [(0, 50)], mode='constant')
        for j in range(50):
            arrCountDispCoeff[i] = arrDispCoeffRng[j]
            arrCost[j] = calCostFunc(frame_data, kaxis, maxDispOrders, arrCountDispCoeff, start_depth, end_depth)
        for k in range(50):
            idx = np.argsort(arrCost)
            firstMin = arrDispCoeffRng[idx[0]]
            secondMin = arrDispCoeffRng[idx[1]]
            arrDispCoeffRng[50 + k] = (firstMin + secondMin) / 2.0
            arrCountDispCoeff[i] = arrDispCoeffRng[50 + k]
            arrCost_append[0] = calCostFunc(frame_data, kaxis, maxDispOrders, arrCountDispCoeff, start_depth, end_depth)
            arrCost = np.concatenate((arrCost, arrCost_append))
        MinI = np.argmin(arrCost)
        arrCountDispCoeff[i] = arrDispCoeffRng[MinI]
    return arrCountDispCoeff, kaxis

""" """
def calCostFunc(frame_data, kaxisCu, maxDispOrders, arrCountDispCoeff, start_depth, end_depth):
    frameData_dispComp = compDisPhase(frame_data, kaxisCu, maxDispOrders, arrCountDispCoeff)
    tmp = fft(frameData_dispComp)
    OCT = np.square(np.absolute(tmp))
    roiOCT = OCT[:, start_depth - 1:end_depth - 1]
    sum_of_oct = np.sum(roiOCT)
    normOCT = np.divide(roiOCT, sum_of_oct)
    log_oct = np.log(normOCT)
    entropy = np.multiply(normOCT, log_oct)
    entropy = np.multiply(entropy, -1)
    cost = np.sum(entropy)
    return cost

def compDisPhase(frame, kaxisCu, maxDispOrders, arrCountDispCoeff):
    [scanPts, linePerFrame] = frame.shape
    # kLinear = np.linspace(-1, 1, linePerFrame)
    # kaxis = np.tile(kLinear, [scanPts, 1])
    frameDisComp = np.copy(frame)
    # print(arrCountDispCoeff, maxDispOrders)
    for i in range (0, maxDispOrders-1):
        kax = np.power(kaxisCu, (i+1))
        expterm = np.multiply(1j, arrCountDispCoeff[i])
        expterm = np.multiply(expterm, kax)
        frameDisComp = np.multiply(frameDisComp, np.exp(expterm))
    return frameDisComp


# """ """
# def compDisPhase(frame_data, kaxisCu, maxDispOrders, arrCountDispCoeff):
#     frameData_dispComp = frame_data
#     [size1, size2] = frame_data.shape
#     power_result = np.zeros((size1, size2))
#     mul_result = np.zeros((size1, size2))
#     complex_result = np.zeros((size1, size2), dtype=np.complex_)
#     exp_result = np.zeros((size1, size2), dtype=np.complex_)
#     for i in range(maxDispOrders - 1):
#         if arrCountDispCoeff[i] == 0:
#             continue
#         power_result = np.power(kaxisCu, i + 2)
#         mul_result = np.multiply(power_result, arrCountDispCoeff[i])
#         complex_result = np.multiply(mul_result, 1j)
#         exp_result = np.exp(complex_result)
#         frameData_dispComp = np.multiply(frameData_dispComp, exp_result)
#     return frameData_dispComp

""" """
def reorderBscan(proc, m, axis=1): # default axis = 1 -> equivalent to fliplr(m)
    volReorder = np.copy(proc)
    m2 = np.mulitply(2, m)
    for frameNum in range(0, proc.shape[2]):
        if (frameNum+1) % (m2) == 0 or (frameNum+1) % (m2) > m:
            volReorder[:,:,frameNum] = np.flip(proc[:,:,frameNum], axis)
    return volReorder

# X = np.array([
# [[1,1,1], [2,2,2]],
# [[3,3,3], [4,4,4]],
# [[5,5,5], [6,6,6]],
# [[7,7,7], [8,8,8]],
# ])
# print(X) # reorderBscan(X, 3)[2,1,2])
# print()
# print(1, reorderBscan(X, 1)) #mkay
# print()
# print(2, reorderBscan(X, 2)) #??
# print()
# print(3, reorderBscan(X, 3)) #??
# print()
# print(4, reorderBscan(X, 4)) #??
# print()
# print(5, reorderBscan(X, 5)) #??
# print()
# print(np.flip(X))
