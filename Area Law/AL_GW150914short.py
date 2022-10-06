#Library Imports
from __future__ import division, print_function
from pylab import *
import numpy as np
import arviz as az
import pandas as pd
import seaborn as sns
import h5py
import ringdown
import matplotlib.pyplot as plt
import scipy
import random
import bilby
from bilby.core.prior import Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
from gwpy.timeseries import TimeSeries
import pycbc.catalog
import pycbc.waveform

sns.set_context('notebook')
sns.set_palette('colorblind')

#Input Parameters - To be updated for each event
G1 = pycbc.catalog.catalog.get_source(source='gwtc-1')
E1 = G1['GW150914-v3']
h1_ringdown = 'H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5'
l1_ringdown = 'L-L1_GWOSC_16KHZ_R1-1126259447-32.hdf5'

right_ascension = 1.95
declination = -1.27
psi_in = 0.82
ph = 0.0
thet = 2.55
tilt1 = 1.65
tilt2 = 1.74
phi12 = 3.15
phijl = 3.24

Ascale = 5e-21
Mf_min = 35.0
Mf_max = 140.0
M_est = 70.0
chi_est = 0.7

tgps = E1['GPS']
M1_min = E1['mass_1_source'] + E1['mass_1_source_lower']
M1_max = E1['mass_1_source'] + E1['mass_1_source_upper']
M2_min = E1['mass_2_source'] + E1['mass_2_source_lower']
M2_max = E1['mass_2_source'] + E1['mass_2_source_upper']
ld = E1['luminosity_distance']

#Functions used in the code
def norm(arr1,arr2):
    diff = arr1[1] - arr1[0]
    func = scipy.interpolate.interp1d(arr1,arr2)
    nor = 0
    for i in range(0,(len(arr1)-1)):
        diff1 = diff/4
        p1 = func(arr1[i])
        p2 = func(arr1[i]+(diff1))
        p3 = func(arr1[i]+(2*diff1))
        p4 = func(arr1[i]+(3*diff1))
        p5 = func(arr1[i+1])
        nor = nor+((2*diff)/45)*((7*p1)+(32*p2)+(12*p3)+(32*p4)+(7*p5))
    return nor 

def read_strain(file, dname):
    with h5py.File(file, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
    
        dt = T/len(h)
    
        raw_strain = ringdown.Data(h, index=t0 + dt*arange(len(h)), ifo=dname)
        
        return raw_strain

def next_pow_two(x):
    y = 1
    while y < x:
        y = y << 1
    return y

def waveform_model(time_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
         a_1, a_2, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, 
         **waveform_kwargs):
    """ Modelling the inspital part of the waveform in time domain """
    
    start_frequency = minimum_frequency = waveform_kwargs.get('minimum_frequency', 20.0)
    #maximum_frequency = waveform_kwargs.get('maximum_frequency', frequency_array[-1])
    reference_frequency = waveform_kwargs.get('reference_frequency', 50.0)
    
    start_time = minimum_time = waveform_kwargs.get('minimum_time', time_array[0])
    end_time = maximum_time = waveform_kwargs.get('maximum_time', time_array[-1])
    
    waveform_dictionary = dict(spin_order=-1, tidal_order=-1,
        phase_order=-1, amplitude_order=0)
    
    m1 = mass_1 * bilby.core.utils.solar_mass
    m2 = mass_2 * bilby.core.utils.solar_mass
    
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=m1, mass_2=m2,
        reference_frequency=reference_frequency, phase=phase)
    
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0
    
    hpt, hct = pycbc.waveform.get_td_waveform(approximant="NRSur7dq4",
                                        mass1=mass_1,
                                 mass2=mass_2,
                                 a1= a_1, a2= a_2,
                                 delta_t=1.0/4096,
                                 f_lower=20)
    
    #h_plus = np.zeros_like(time_array, dtype=complex)
    #h_cross = np.zeros_like(time_array, dtype=complex)
    
    if len(hpt.data.data) > len(time_array):
        h_plus = hpt.resize(len(time_array))
        h_cross = hct.resize(len(time_array))
    else:
        h_plus = hpt.cyclic_time_shift(hpt.start_time)
        h_cross = hct.cyclic_time_shift(hct.start_time)
    
    return dict(plus = h_plus, cross = h_cross)

#Logger and Interferometers
logger = bilby.core.utils.logger
trigger_time = tgps
detectors = ["H1", "L1"]
maximum_frequency = 1024
minimum_frequency = 75
roll_off = 0.4  #Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  #Analysis segment duration
post_trigger_duration = 0  #Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)
    
#Priors
prior = bilby.gw.prior.PriorDict()
prior["geocent_time"] = bilby.core.prior.Uniform(trigger_time - 0.1, trigger_time + 0.1, name="geocent_time")
prior["mass_1"] = bilby.core.prior.Uniform(M1_min, M1_max, r"m1")
prior["mass_2"] = bilby.core.prior.Uniform(M2_min, M2_max, r"m2")
prior["a_1"] = bilby.core.prior.analytical.Uniform(name='a_1', minimum=0, maximum=0.99)
prior["a_2"] = bilby.core.prior.analytical.Uniform(name='a_2', minimum=0, maximum=0.99)
prior["luminosity_distance"] = ld
prior["ra"] = right_ascension
prior["dec"] = declination
prior["psi"] = psi_in
prior["phase"] = ph
prior["theta_jn"] = thet
prior["tilt_1"] = tilt1
prior["tilt_2"] = tilt2
prior["phi_12"] = phi12
prior["phi_jl"] = phijl

duration = 4
sampling_frequency = 4096

#Defining Likelihood and running fit
waveform_generator = bilby.gw.WaveformGenerator(duration = duration, sampling_frequency = sampling_frequency, time_domain_source_model = waveform_model, start_time = start_time)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifo_list,waveform_generator, priors=prior)

result = bilby.run_sampler(likelihood, prior, sampler="dynesty", nlive=25, npoints=50, 
                           sample='unif',
                           conversion_function=bilby.gw.conversion.generate_all_bbh_parameters)

#Ringdown Analysis
h_raw_strain = read_strain(h1_ringdown, 'H1')
l_raw_strain = read_strain(l1_ringdown, 'L1')

longest_tau = ringdown.qnms.get_ftau(M_est, chi_est, 0, l=2, m=2)[1]
highest_drate = 1/ringdown.qnms.get_ftau(M_est, chi_est, 1, l=2, m=2)[1]
print('The damping rate of the second tone is: {:.1f} Hz'.format(highest_drate))
print('The time constant of the first tone is: {:.1f} ms'.format(1000*longest_tau))

T = 10*longest_tau
srate = next_pow_two(2*highest_drate)
print('Segment of {:.1f} ms at sample rate {:.0f}'.format(1000*T, srate))

#Choosing Ringdown model and conditioning data
fit = ringdown.Fit(model='mchi', modes=[(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)])  # use model='ftau' to fit damped sinusoids instead of +/x polarized GW modes
fit.add_data(h_raw_strain)
fit.add_data(l_raw_strain)
fit.set_target(tgps, ra=right_ascension, dec=declination, psi=psi_in, duration=T) #check if ra dec and psi are correct
fit.condition_data(ds=int(round(h_raw_strain.fsamp/srate)), flow=1/T)
fit.compute_acfs()
wd = fit.whiten(fit.analysis_data)

#Setting Priors and running fit
fit.update_prior(A_scale=Ascale, M_min=Mf_min, M_max=Mf_max, flat_A=True)
fit.run(draws=1000,random_seed=1234)

#Calculating final area 
m = np.array(fit.result.posterior.M)
spin = np.array(fit.result.posterior.chi)
mass = np.array([])
a = np.array([])
for i in range(0,4):
    mass = np.append(mass,m[i])
    a = np.append(a,spin[i])

G = 6.67 * (10**(-11))
c = 9 * (10**9)
area = [0 for i in range(0,len(mass))]
for i in range(0,len(mass)): 
    temp = 8*np.pi*(((G*mass[i])/(c**2))**2)*(1+((1-(a[i]**2))**(1/2)))
    area[i] = temp
area_f = np.array(area)

#Initial Areas
mass1 = np.array(result.posterior.mass_1_source)
mass2 = np.array(result.posterior.mass_2_source)
a1 = np.array(result.posterior.a_1)
a2 = np.array(result.posterior.a_2)

area1 = np.array([8*np.pi*(((G*mass1[i])/(c**2))**2)*(1+((1-(a1[i]**2))**(1/2))) for i in range(0,len(mass1))])
area2 = np.array([8*np.pi*(((G*mass2[i])/(c**2))**2)*(1+((1-(a2[i]**2))**(1/2))) for i in range(0,len(mass1))])
area_i = np.array([(area1[i] + area2[i]) for i in range(len(area1))]) #total initial area

#Gaussian KDES - Check how these look with different smoothening parameters and bandwidths
f2 = scipy.stats.gaussian_kde(area_i,bw_method='silverman') #KDE for initial area
f1 = scipy.stats.gaussian_kde(area_f,bw_method='silverman') #KDE for final area

N = 500 #Number of analysis points
start = np.min(area_i)/2
finish = np.max(area_f)*2
u = np.linspace(start,finish,N)

#Integral for A_f/A_i ratio
s = 0.0
f = np.max(area_f)/np.min(area_i) 
I = [0.0 for i in range(0,N)]
z = np.linspace(s,f,N)
x = u
h = (u[1]-u[0])/len(x)

for i in range(0,len(z)):
    for j in range(0,(len(x)-1)):
        h1 = h/4
        k1 = f1.evaluate(x[j]*z[i]) * f2.evaluate(x[j]) * x[j]
        k2 = f1.evaluate((x[j]+(h1))*z[i]) * f2.evaluate(x[j]+(h1)) * (x[j]+(h1))
        k3 = f1.evaluate((x[j]+(2*h1))*z[i]) * f2.evaluate(x[j]+(2*h1)) * (x[j]+(2*h1))
        k4 = f1.evaluate((x[j]+(3*h1))*z[i]) * f2.evaluate(x[j]+(3*h1)) * (x[j]+(3*h1))
        k5 = f1.evaluate(x[j+1]*z[i]) * f2.evaluate(x[j+1]) * x[j+1]
        I[i] = I[i] + ((2*h)/45)*((7*k1)+(32*k2)+(12*k3)+(32*k4)+(7*k5))

z = z - 1
# Normalising dA/A distribution
temp = np.array([])
for i in I:
    temp = np.append(temp,i)
I = temp
normal = norm(z,I)
I = I/normal

#Plotting dA/A and exporting data to a file
plt.plot(z,I)
plt.savefig('ratio.png')

x = np.array([z,I])
np.savetxt('zI.txt',x)
np.savetxt('mass1.txt',mass1)
np.savetxt('mass2.txt',mass2)
np.savetxt('a1.txt',a1)
np.savetxt('a2.txt',a2)
np.savetxt('massf.txt',mass)
np.savetxt('af.txt',a)
