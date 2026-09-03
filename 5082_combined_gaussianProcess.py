import numpy as np
import matplotlib.pyplot as plt
import ellc
import edmcmc as edm
import corner
import pandas as pd
from matplotlib import gridspec
from matplotlib.pyplot import cm
import batman
from astropy.io import ascii
import math
import os

from scipy.optimize import least_squares
from scipy.stats import linregress
import jax
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels
import concurrent.futures
jax.config.update('jax_platforms', 'cpu')
import multiprocessing as mp
settings = np.seterr(over="ignore")

plot_params = {
    "axes.labelsize": 18,
    "axes.labelpad": 9,
    "axes.titlesize": 20,
    "axes.linewidth": 2,
    "axes.labelweight": 3,
    "axes.titleweight": 3,
    "font.size": 15,
    "legend.fontsize": 15,
    "lines.linewidth": 2,
    "xtick.major.width": 2,
    "xtick.minor.width": 1,
    "xtick.major.size": 8,
    "xtick.minor.size": 5,
    "xtick.major.pad": 5,
    "xtick.labelsize": 15,
    "xtick.minor.visible": True,
    "xtick.direction": "in",
    "xtick.top": True,
    "ytick.major.width": 2,
    "ytick.minor.width": 1,
    "ytick.major.size": 8,
    "ytick.minor.size": 5,
    "ytick.major.pad": 7,
    "ytick.labelsize": 15,
    "ytick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.right": True,
    "legend.frameon": True,
    "legend.loc": "upper right",
    #'text.usetex': True,
    #'text.latex.preamble': '\usepackage{helvet}\usepackage[T1]{fontenc}\usepackage{sfmath}',
    #'font.sans-serif': "Helvetica",
    #'font.family': "sans-serif",
    "ps.usedistiller": "xpdf",
    "savefig.dpi": 300,
    "figure.figsize": [7, 7],
}

plt.rcParams.update(plot_params)

# adding noise to transit
def std_dev(arr_time):
    t_or_f = arr_time < -0.25
    arr_mask_1 = arr_time[t_or_f]
    masking2 = arr_time > 0.175
    arr_mask_2 = arr_time[masking2]

    # finding the values where the noise starts
    val1 = arr_mask_1[-1]
    val2 = arr_mask_2[0]

    # index where it occurs
    idx1 = np.where(arr_time == val1)
    idx2 = np.where(arr_time == val2)
    return idx1, idx2


def unpack_file(filepath):
    df = pd.read_csv(filepath, comment="#")
    for col in ("ccfjdsum", "ccfrvmod", "dvrms"):
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: {col}")

    time_rv_unsort = df["ccfjdsum"].values.astype(float)  # times in days
    data_rv_unsort = df["ccfrvmod"].values.astype(float)  # observed RV in km/s
    err_rv_unsort = df["dvrms"].values.astype(float)  # RV uncertainties in km/s

    idx_1 = np.argsort(time_rv_unsort) #tinyGP requires sorted data
    time_rv = time_rv_unsort[idx_1]
    data_rv = data_rv_unsort[idx_1]
    err_rv = err_rv_unsort[idx_1]
    
    data_rv -= np.median(data_rv)
    return time_rv, data_rv, err_rv

def sma_star_units(sma, rstar):
    return (sma*rstar) #rstar in units of Rsun

## Input files and output dir

#Transit (cleaned)
directory = '/home/nadja/Documents/UWMadison/Research/TOI5082/'

transit_file = ascii.read(directory+"tic437011608flattened-2min.csv")
time = transit_file["Time (BJD-2457000)"] 
transit_flux = transit_file["Flattened Flux"]
transit_time = time + 2457000


#RVs
first_rv = (directory + "NEID_TOI5082_RM_Event202502.csv")
second_rv = (directory + "2026-01-20_TOI5082.csv")

time_obs_1, rv_data_1_RAW, rv_err_1 = unpack_file(first_rv)
time_obs_2, rv_data_2_RAW, rv_err_2 = unpack_file(second_rv)
write_chains = True
thin_n = 20

#mcmc
nlink = 75000
nburnin = 10000 

#initial guesses
t0_bjd = 2459508.8190741274
rp_rs = (10 ** (-3) * 0.93**2) ** 0.5  # stellar radius units
sma = 11.8  #RATIO
RSTAR = 0.93
a_calc = sma_star_units(sma, RSTAR)
vsini = 7
lambda_guess = 0
incl = 87
r_1 = 1 / sma
r_2 = rp_rs * r_1
orbital_period = 4.2403567  # orbital_period
q_guess = 0.00002109

shape = 'sphere'
e_val = 0
periastron = 90
coef_1 = 0.1
coef_2 = 0.3

fs_guess = np.sqrt(e_val)*math.sin(np.deg2rad(periastron))
fc_guess  = np.sqrt(e_val)*math.cos(np.deg2rad(periastron))
p_star_guess = 1.743589

# GP initial guesses 
amplitude = 0.0002
scale = 0.1

#num_events_1 = int(np.round((rv_t0-trans_t0)/orbital_period))
#num_events_2 = int(np.round((rv_t0_2 - trans_t0)/orbital_period))
#K_guess = 0.5*(np.nanmax(rv_data_1)-np.nanmin(rv_data_1))

labels=["a_over_R", "radius_planet", "vsini", "obliquity", "t0_trans", "incl", "period", 'fs', 'fc', 'c1', 'c2',  'amp', 'scale', 'slope_1', 'intercept_1', 'slope_2', 'intercept_2']

#systemic offset
def systemic_offset(time, data, err):
    i_rad = np.radians(incl)
    rsum = (r_1 + r_2)
    val = rsum / max(1e-12, np.sin(i_rad))
    if val >= 1.0:
        transit_duration_days = 0.2
    else:
        transit_duration_days = orbital_period / np.pi * val
    transit_half_phase = (transit_duration_days / 2.0) / orbital_period

    phases_for_mask = ((time - t0_bjd) / orbital_period + 0.5) % 1.0 - 0.5 #change
    in_transit_mask = np.abs(phases_for_mask) < transit_half_phase
    out_of_transit_mask = ~in_transit_mask

    if out_of_transit_mask.sum() < 3:
        out_of_transit_mask = np.ones_like(out_of_transit_mask, dtype=bool)
    weights = 1.0 / (err**2) #change

    gamma_weighted = np.sum(weights[out_of_transit_mask] * data[out_of_transit_mask]) / np.sum(weights[out_of_transit_mask]) #change
    data = data - gamma_weighted #change
    return data
'''
rv_data_1 = systemic_offset(time_obs_1, rv_data_1_RAW, rv_err_1)
rv_data_2 = systemic_offset(time_obs_2, rv_data_2_RAW, rv_err_2)'''

rv_data_1 = rv_data_1_RAW
rv_data_2 = rv_data_2_RAW

#TESS standard dev
phase_trans = ((transit_time - t0_bjd) / orbital_period) - np.round(
    (transit_time - t0_bjd) / orbital_period
)

i, j = std_dev(phase_trans)

idx1 = i[0][0]
idx2 = j[0][0]

noise_arr = np.concatenate((transit_flux[:idx1], transit_flux[idx2:]), axis=0)
error = np.nanstd(noise_arr)
trans_err = np.full_like(transit_flux, error)

## bg trend
def lin_trend(time, data):
    time_centered = time - np.median(time)
    slope, intercept, r_val, p_val, std_err = linregress(time_centered, data)
    return slope, intercept

#linear background trend + defining GP
slope_1, intercept_1 = lin_trend(time_obs_1, rv_data_1)
slope_2, intercept_2 = lin_trend(time_obs_2, rv_data_2)

def build_gp(params, time):
    kernel = kernels.quasisep.Matern32(scale=jnp.exp(params['log_scale']), sigma=jnp.exp(params['log_amp']))
    return GaussianProcess(kernel, time, diag=jnp.exp(params["log_diag"]))

# Commented here is the edmcmc I made for batman. Dunno if this helps
def lnprior(p):
    r_star_sma, rplanet, vsini_guess, obliquity, t0_trans, inc, period, fs, fc, c_1, c_2, amp, scale, a, c, m, d  = p
    r_1 = 1/r_star_sma
    r_2 = r_1 * rplanet
    if t0_trans < t0_bjd - 1 or t0_trans >  t0_bjd+1:
        return -np.inf
    '''if period > orbital_period + 0.0001 or period < orbital_period - 0.0001:
        return -np.inf
    if (r_1+r_2) > 1 or (r_1+r_2) < 0:
        return -np.inf'''
    if rplanet <= 0 or rplanet >= 1:
        return -np.inf
    if inc < 0 or inc > 90:
        return -np.inf
    if fs <=-1 or fs >= 1:
        return -np.inf
    if fc <=-1 or fc >= 1:
        return -np.inf
    if c_1 > 1 or c_1 <0:
        return -np.inf
    if c_2 > 1 or c_2 < 0:
        return -np.inf
    if obliquity > 180 or obliquity < -180:
        return -np.inf
    if vsini_guess < 0 or vsini_guess > 20:
        return -np.inf
    w_peri = np.arctan2(fs, fc)
    ecc = fs**2 +fc**2
    if ecc >=0.975:
        return -np.inf
    if not np.isfinite(ecc):
        return -np.inf
    b = np.abs(r_star_sma * np.cos(np.deg2rad(inc)) * (1-ecc**2)/(1+ecc*np.sin(w_peri)))
    if b < 0 or b > 1:
        return -np.inf
    if not (0 < amp < 0.005):
        return -np.inf
    if not (0 < scale < 0.21):
        return -np.inf

    p_star = (3 * np.pi * pow(r_star_sma,3))/(497.582 * period**2)

    mu_p_star = 1.743589  # yea change this
    sigma_p = 0.1

    # Calculate individual priors
    prior_pstar = (
        np.log(1.0 / (np.sqrt(2 * np.pi) * sigma_p))
        - 0.5 * (p_star - mu_p_star) ** 2 / sigma_p**2
    )
    return prior_pstar

def trans_loglikelihood(p, trans_time, trans_obs, err):
    r_star_sma, rplanet, vsini_guess, obliquity, t0_trans, inc, period, fs, fc, c_1, c_2= p
    ecc = fs **2 + fc **2 
    w_peri = np.arctan2(fs, fc)

    batman_params = batman.TransitParams()
    batman_params.t0 = t0_trans # time of inferior conjunction
    batman_params.rp = rplanet  # planet radius (in units of stellar radii)
    batman_params.a = r_star_sma  # semi-major axis (in units of stellar radii)
    batman_params.inc = inc  # orbital inclination (in degrees)
    batman_params.per = period  # orbital period
    batman_params.ecc = ecc  # eccentricity

    batman_params.w = np.rad2deg(w_peri)  # longitude of periastron (in degrees)
    batman_params.u = [c_1, c_2]  # limb darkening coefficients [u1, u2]
    batman_params.limb_dark = "quadratic"  # limb darkening model

    try: 
        m = batman.TransitModel(batman_params, trans_time)
        model = m.light_curve(batman_params)
    except Exception:
        return -np.inf
    
    norm = np.sum(np.log(2*np.pi*err**2))
    chisq = np.sum((trans_obs - model) ** 2 / err**2)

    trans_loglikelihood = -0.5 * (chisq+norm)
    return trans_loglikelihood

def rv_loglikelihood(p, trend, time, rv_obs, rv_err):
    (r_star_sma, rplanet, vsini_guess, obliquity, t0_trans, inc, 
     period, fs, fc, c_1, c_2, amp, scale_val) = p
    (slope_m, intercept) = trend
    r1 = 1 / r_star_sma
    r2 = rplanet * r1
    ## GP inclusion

    semi_major_axis = sma_star_units(r_star_sma, RSTAR)
    rv_model, _ = ellc.rv(
        time,
        t_zero=t0_trans,
        period=period,
        lambda_1=obliquity,
        radius_1=r1,
        radius_2=r2,
        incl=inc,
        f_s=fs,
        f_c=fc,
        a=semi_major_axis,
        shape_1="sphere",
        shape_2="sphere",
        vsini_1=vsini_guess,
        flux_weighted=True,
        sbratio=0,
        q=q_guess,
        verbose=0
        )
    
    time_centered = time - np.median(time)
    model_slope = slope_m * time_centered + intercept

    base = rv_model + model_slope
    #norm = np.sum(np.log(2*np.pi*rv_err**2))
    #chisq = np.sum((rv_obs - base) ** 2 / rv_err**2)

    #call gp func instead of of loglikelihood
    theta = {
        "log_amp": jnp.log(amp),    
        "log_scale": jnp.log(scale_val),      
        "log_diag": jnp.log(rv_err**2)} 
    
    gp = build_gp(theta, time_centered)
    logl = gp.log_probability((rv_obs - base))
    
    #rv_loglikelihood = -0.5 * (chisq+norm)
    return logl

def lnprob(p, time_rv, time_rv_2, time_trans, rv_obs_1, rv_obs_2, trans_obs, rv_err_1, rv_err_2, trans_err):
    penal_pstar = lnprior(p)
    if not np.isfinite(penal_pstar):
        return -np.inf
    p_orbit = p[0:11]    
    p_rv = p[0:13] 
    a, c, b, d = p[13:17] 
    
    rv_logp_1 = rv_loglikelihood(p_rv, (a, c), time_rv, rv_obs_1, rv_err_1) 
    if not np.isfinite(rv_logp_1):
        return -np.inf
        
    rv_logp_2 = rv_loglikelihood(p_rv, (b, d), time_rv_2, rv_obs_2, rv_err_2) 
    if not np.isfinite(rv_logp_2):
        return -np.inf
        
    trans_logp = trans_loglikelihood(p_orbit, time_trans, trans_obs, trans_err)
    if not np.isfinite(trans_logp):
        return -np.inf
    
    total_logp = rv_logp_1 + trans_logp + penal_pstar + rv_logp_2
    return (float(total_logp))

## IF USING THIS METHOD PUT EVERYTHING IN THE FUNCTION ALSO SET THE METHOD INSIDE THE FUNCTION TO NOT CRASH THE PROGRAM.
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    p0 = [sma, rp_rs, vsini, lambda_guess, t0_bjd, incl, orbital_period, 
          fs_guess, fc_guess, coef_1, coef_2, amplitude, scale, 
          slope_1, intercept_1, slope_2, intercept_2]
    
    # Pass the explicit pool directly to edmcmc instead of just an integer
    out = edm.edmcmc(
    lnprob, p0, [0.01, 0.001, 0.1, 1, 0.00001, 0.001, 0.00001, 0.05, 0.05, 0.001, 0.001, 0.001, 0.001, 0.000001, 0.001, 0.000001, 0.001], #probably have to tighten constraints in the scale as to not blow it up
    args=(time_obs_1, time_obs_2, transit_time, 
    rv_data_1_RAW, rv_data_2_RAW, transit_flux, 
    rv_err_1, rv_err_2, trans_err),
    nwalkers=100,  
    nlink=nlink,
    nburnin=nburnin,
    m1mac=False,
    ncores=8,)
        
    samples_for_outputs = out.get_chains(nthin=thin_n, nburnin=nburnin, flat=True)

    if write_chains:
        np.savez(
            directory + 'combined_chains_75000_GP.npz',
            thinflatchains=samples_for_outputs,
            lastpos=out.lastpos,
            nwalkers=out.nwalkers,
            npar=out.npar,
            nburnin=out.nburnin,
            thin_n=thin_n,
            nlink=out.nlink,
            labels=np.array(labels, dtype=object),
    )

    thinned_all_samples = out.get_chains(nthin=thin_n, nburnin=nburnin, flat=False)
    np.savez(
            directory + 'combined_allchains_75000_GP.npz',
            allchains = thinned_all_samples,
    )

    gelmanrubinmetrics = out.gelmanrubin()
    with open("gelmann_rubin_combined_GP.txt", "w") as f:
        for i in range(len(gelmanrubinmetrics)):
            print('Parameter number ' + str(i+1) + ' (' + labels[i]+') has a Gelman-Rubin statistic of '
                + str(gelmanrubinmetrics[i])+'.', file=f)

    with open(directory+ "median_params.txt", "w") as f:
        for i in range(len(labels)):
            print(f'{labels[i]}: {np.median(out.flatchains[:, i])}+/-{np.nanstd(out.flatchains[:,i])} \n', file=f)
