# Autofit_Experiments_2023-02.py
#
# Test data filtering and MLE method for determining the maximum
# David Lister
# February 2023
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, odr, signal, optimize

fname = "H1235-L_refl.csv"  # Should be around 940 nm thick

data = np.loadtxt(fname, delimiter=',')
data = data.transpose()
data[0] = data[0] * 1e-9

faketime = np.linspace(0, 1, num=data.shape[1])

N = len(faketime)
yf = fft.fft(data[1])
xf = fft.fftfreq(N, faketime[1] - faketime[0])

##plt.loglog(xf, 2.0/N * np.abs(yf), 'x')
##plt.show()


# These filter values should be configurable
yf[xf < 4] = 0
yf[xf > 200] = 0

filtered = np.real(fft.ifft(yf))

yf = fft.fft(data[1])
mask = xf > 4
mask = mask & (xf < 200)
yf[mask] = 0

residual = np.real(fft.ifft(yf))

plt.title("Raw vs Filtered reflectance spectra")
plt.xlabel("Wavelength (m)")
plt.ylabel("Arbirary")
plt.plot(data[0], residual, label="Residual")
plt.plot(data[0], data[1], label="Raw")
plt.plot(data[0], filtered, label="Filtered")
plt.legend()
plt.show()

def model(B, window_poly, x):
    """
    B[0] is proportional to thickness, and other stuff. Eqn 3.5
    B[1] is a phase factor
    """
##    print(B, window_poly, len(x))
    return np.polyval(window_poly, x) * np.cos((4 * np.pi / x) * B[0] * 1.6  + B[1])

def modelB(B, x):
##    print(B, window_poly, len(x))
    return B[0] * np.cos((4 * np.pi / x) * B[1] * 1.6)


def rsquared(a, b):
    return np.sqrt(np.sum((a - b)**2))

def lonenorm(a, b):
    return np.sum(np.abs(a - b))


mask = data[0] > 400e-9
mask = mask * (data[0] < 800e-9)

snipx = data[0][mask]
snipy = filtered[mask]


mydata = odr.RealData(data[0], filtered)
m = odr.Model(modelB)
guess_list = [600e-9, 900e-9, 950e-9, 1000e-9]
for guess in guess_list:
    fit = odr.ODR(mydata, m, beta0=[0.1, guess, 0])
    out = fit.run()
    test = modelB(out.beta, data[0])
    plt.plot(data[0], test, alpha=0.25, label = f"Guess: {guess:3.2e}m Fit: {out.beta[1]:3.3e}m")

plt.title("Many metastable fits")
plt.xlabel("Wavelength (m)")
plt.ylabel("Arbitary")
plt.plot(data[0], filtered, label="Filtered")
plt.legend()
plt.show()


do_raw_fitting_surface = False
if do_raw_fitting_surface:
    dlst = np.linspace(100e-9, 1100e-9, 1001)
    qlist = []
    qlist_lone = []
    mydata = odr.RealData(snipx, snipy)
    m = odr.Model(lambda B, x: modelB([B[0], d], x))

    for d in dlst:    
        fit = odr.ODR(mydata, m, beta0=[0.1, 700e-9, 0])
        out = fit.run()

        test = modelB(out.beta, snipx)
        quality = 1/rsquared(test, snipy)
        qlist.append(quality)
        qlist_lone.append(1/lonenorm(test, snipy))
        print(d, quality)


    ##plt.plot(snipx, modelB([0.1, 922e-9], snipx))
    ##plt.plot(snipx, snipy)
    ##plt.show()
        
    plt.title("Model fitting surface")
    plt.xlabel("Thickness (m)")
    plt.ylabel("$||\\Delta||_2^{-1}$", rotation=0)
    plt.plot(dlst, qlist)
    ##plt.plot(dlst, qlist_lone)
    plt.show()

snipy_abs = np.abs(snipy)
peakx_idx, props = signal.find_peaks(snipy_abs, width=21)
peakx = snipx[peakx_idx]
peaky = snipy_abs[peakx_idx]

poly = np.polyfit(peakx, peaky, 3)

plt.title("Envelope Function")
plt.xlabel("Wavelength (m)")
plt.ylabel("Arbitrary")
plt.plot(peakx, peaky, 'rx')
plt.plot(snipx, snipy_abs, label="Absolute value of spectrum")
plt.plot(snipx, np.polyval(poly, snipx), label="Polynomial fit")
plt.legend()
plt.show()


m = odr.Model(lambda B, x: model(B, poly, x))

mydata = odr.RealData(snipx, snipy)
fit = odr.ODR(mydata, m, beta0=[900e-9, 0])
out = fit.run()
out.pprint()

##plt.plot(snipx, snipy)
##plt.plot(snipx, model(out.beta, poly, snipx))
##plt.show()


dlst = np.linspace(100e-9, 1800e-9, 1701)
qlist = []
qlist_lone = []
plot = False

do_iteration = False
if do_iteration:
    for d in dlst:
        m = odr.Model(lambda B, x: model([d, B[0]], poly, x))
        
        mydata = odr.RealData(snipx, snipy)
        fit = odr.ODR(mydata, m, beta0=[0])
        out = fit.run()

        test = model([d] + list(out.beta), poly, snipx)
        if plot:
            plt.plot(test)
            plt.plot(snipy)
            plt.show()
        quality = 1/rsquared(test, snipy)
        qlist.append(quality)
        qlist_lone.append(1/lonenorm(test, snipy))
        print(d, quality)

    plt.title("Envelope model surface")
    plt.xlabel("Thickness (m)")
    plt.ylabel("$||\\Delta||_2^{-1}$", rotation=0)
    plt.plot(dlst, qlist, label="L2")
    ##plt.plot(dlst, qlist_lone, label="L1")
    ##plt.legend()
    plt.show()

    kernel = lambda p, c, x: (-p/c**2) * (x - c) * (x + c)

    v = 20
    peak = 0.3
    x = np.linspace(-v, v, 2*v + 1)
    k = kernel(peak, v, x)

    plt.plot(x, k)
    plt.show()

    conv = np.convolve(k, qlist, mode='same')

    plt.title("Convolved model surface")
    plt.xlabel("Thickness (m)")
    plt.ylabel("Arbitrary")
    plt.plot(dlst, conv)
    plt.show()

    conv = list(conv)
    idx = conv.index(max(conv))
    thickness = dlst[idx]
    print(f"\n\nThe thickness is: {thickness:1.3e} nm")

else:
    thickness = 949e-9

yf = fft.fft(data[1])
mask = xf > 4
mask = mask & (xf < 200)
yf[mask] = 0

residual = np.real(fft.ifft(yf))

##plt.plot(data[0], residual, label="res")
##plt.plot(data[0], data[1], label="Raw")
##plt.plot(snipx, model([thickness, 0], poly, snipx))
##plt.show()

def fine_model(arr):
    thickness = arr[0]
    phase = arr[1]
    fit = model([thickness, phase], poly, snipx)
    err = lonenorm(fit, snipy)
    return err


# This doesn't really do much
res = optimize.minimize(fine_model, (thickness, 0), method="trust-constr")
print("\n\n")
print(res)

fit = model(res.x, poly, snipx)

##plt.plot(snipx, fit)
##plt.plot(snipx, snipy)
##plt.show()

offset = list(data[0]).index(snipx[0])
residual[offset:offset + len(snipx)] += fit 

plt.title("Final fit compared to raw data")
plt.plot(data[0], data[1], label="Raw")
plt.plot(data[0], residual, label="Residual + Fit")
plt.xlabel("Wavelength (m)")
plt.ylabel("Arbitrary")
plt.legend()
plt.show()

plt.title("Final fit vs filtered data")
plt.xlabel("Wavelength (m)")
plt.ylabel("Arbitrary")
plt.plot(snipx, snipy, label="Filtered")
plt.plot(snipx, fit, label="Fit")
plt.legend()
plt.show()
