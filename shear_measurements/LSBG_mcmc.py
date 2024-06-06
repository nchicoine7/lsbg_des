#Imports

import os
import emcee
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import time
import config
from matplotlib import ticker
from scipy import optimize
import twopoint
from profiley.nfw import TNFW, NFW
import colossus
from colossus.cosmology import cosmology
from colossus.halo import concentration
cosmology.setCosmology('planck15')
from astropy.cosmology import Planck15
from astropy import constants as const
from scipy import stats 

#loading data
cov_red = np.loadtxt('../data/gt_jkcov_red')
theta_red, xi_red, error_red, cross_term_red=np.loadtxt('../data/gt_LSBG_red', unpack=True)
theta_red, xi_red, err_jk_red, cross_term_red = np.loadtxt('../data/gt_LSBG_jackknife_red', unpack=True)
zldist_red = np.loadtxt('../data/zldist_red', unpack=True)

#loading source redshift mean
zs = 0.6312412596033679

#converting from angular scales (arcmin) to radians
theta_rad = theta_red/60*np.pi/180

#size of offset distribution
size_off = 40

#number of walkers, parameters
nwalkers = 20
ndim = 3

#lens redshift range
zlrange = np.linspace(0.005, 0.135, 14)

#function definitions

def hartlap(n=100, m=22):
	"""
	Calculates the Hartlap-Kaufman factor
	n = number of realizations (in our case 100)
	m = number of data entries (in our case 22)
	Returns: Hartlap-Kaufman factor
	""" 
	hartlap_factor = (n - m - 2)/ float(n-1)
	return hartlap_factor

def normalization(mean=None, sigma=None, size=None):
	"""
	Normalizes the offset distribution
	mean = mean of radial offset distribution (arcmin)
	sigma = spread of distribution
	size = size of distribution
	Returns: offset range, offset distribution, normalization factor, normalized offset distribution
	"""
	func_range = np.linspace(mean - 2*sigma, mean + 2*sigma, size) 
	dist = stats.norm.pdf(func_range, mean, sigma) 
	A_norm = np.trapz(dist) 
	dist_norm = dist/A_norm
	return func_range, dist, A_norm, dist_norm

def nfw_generation(mass=None, zl=None, zs=None, scales=None):
	"""
	Constructs an NFW profile based on the halo mass, lens redshift, source redshift, and angular scales 
	mass = halo mass (solar masses)
	zl = lens redshift
	zs = source redshift
	scales = angular scales (radians)
	Returns: angular diameter distance, physical scales (Mpc), NFW profile, sigma_crit factor
	"""
	da = Planck15.angular_diameter_distance(zl) 
	r_mpc = da*scales 
	concentration_xi = concentration.concentration(mass, '200c', zl, 'ishiyama21')
	nfw = NFW(mass, concentration_xi, zl, overdensity=200, background='c') 
	sigma_crit = nfw.sigma_crit([zs])
	return da, r_mpc, nfw, sigma_crit

def subhalo_generation(zlrange=None, zldist_norm = None, mass = None, zs = None, scales = None):
	"""
	Constructs a subhalo model based on the NFW profile and normalized lens redshift distribution
	zlrange = lens redshift range
	zldist_norm = normalized redshift distribution
	mass = subhalo mass
	zs = mean of source redshift distribution
	scales = angular scales (radians)
	Returns: subhalo model
	"""
	gt_list = []
	for i, xi in enumerate(zlrange):
		da, r_mpc, nfw, sigma_crit = nfw_generation(mass=mass, zl=xi, zs=zs, scales=scales)
		esd = nfw.projected_excess(r_mpc) 
		gt = esd/sigma_crit
		gt_norm = gt*zldist_norm[i]
		gt_list.append(gt_norm)
	gt_term1 = np.trapz(np.array(gt_list), axis=0)
	return(gt_term1)

def offset_generation(offset=None, zl=None, width=None, size_off=None):
	"""
	Constructs offset distribution
	offset = mean radial offset (arcmin)
	zl = lens redshift
	width = width factor to determine the spread of offset distribution
	size_off = size of offset distribution
	Returns: offset range, offset distribution, normalized offset distribution
	"""
	da = Planck15.angular_diameter_distance(zl) 
	offset_rad = (offset/60)*(np.pi/180)
	marker_roff = np.array(offset_rad*da)
	sigma_roff = marker_roff/width
	r_off_range, r_off_dist, A_off, r_off_norm = normalization(mean=marker_roff, sigma=sigma_roff, size=size_off, func='off') 
	return r_off_range, r_off_dist, r_off_norm

def hosthalo_generation(zlrange=None, zldist_norm=None,  mass=None, zs=None, scales=None, offset=None, size_off=None, width=None):
	"""
	Constructs a host halo model based on the NFW profile, normalized lens redshift distribution, and normalized offset distribution
	zlrange = lens redshift range
	zldist_norm = normalized lens redshift distribution
	mass = host halo mass (solar mass units)
	zs = mean of source redshift distribution
	scales = angular scales (radians)
	size_off = size of offset distribution
	width = determining width factor for offset distribution
	Returns: host halo model
	"""
	gt_list = []
	for i, xi in enumerate(zlrange):
		da, r_mpc, nfw, sigma_crit = nfw_generation(mass=mass, zl=xi, zs=zs, scales=scales)
		r_off_range, r_off_dist, r_off_norm = offset_generation(offset=offset, zl=xi, width=width, size_off=size_off)
		esd = nfw.offset_projected_excess(np.array(r_mpc), np.array(r_off_range))
		gt = np.reshape(esd/sigma_crit, (size_off, 22))
		gt_norm_roff = gt*r_off_norm[:, np.newaxis]
		gt_norm = gt_norm_roff*zldist_norm[i]
		gt_list.append(gt_norm)
	gt_roff_int = np.trapz(np.array(gt_list), axis=1)
	gt_term2 = np.reshape(np.trapz(gt_roff_int,axis=0), (22,1))
	return gt_term2


def theory(pars):
	"""
	Calculates the total shear profile model
	pars = subhalo mass, host halo mass, mean radial offset
	Returns: total model
	"""
	gt_term1 = subhalo_generation(zlrange=zlrange, zldist_norm=zldist_red, zs=zs, mass=10**pars[0], scales=theta_rad) 
	gt_term2 = hosthalo_generation(zlrange=zlrange, zldist_norm=zldist_red, zs=zs, mass=10**pars[1], scales=theta_rad, offset=pars[2], size_off=size_off, width=3)
	total_model = gt_term1 + gt_term2
	np.savetxt('model_gt', total_model, header = 'Complete Model')
	gt = np.loadtxt('model_gt')
	return gt 

def lnlike(pars, data, cov):
	"""
	Calculates the likelihood of the data given the model parameters
	pars = model parameters (subhalo mass, host halo mass, mean radial offset, etc)
	data = tangential shear measurements
	cov = covariance of tangential shear measurements
	Returns: likelihood
	"""
	invcov = np.linalg.inv(cov)
	theory_it = theory(pars)
	hartlap_factor = hartlap()
	chi2 = np.dot((data-theory_it), np.dot(invcov, (data-theory_it)))*hartlap_factor
	return -0.5 * chi2

def lnprior(pars):
	"""
	Establishes the priors on the parameters for the likelihood function
	pars = parameters (subhalo mass, host halo mass, mean radial offset)
	"""
	mass_term1 = pars[0]
	mass_term2 = pars[1]
	offset = pars[2]
	if ((not 7<mass_term1<12) or (not 10<mass_term2<15) or (not 25<offset<55)): 
		return -np.inf
	else:
		return 0.
		
def lnprob(pars, data, cov):
	"""
	Calculates the value of the prior and the likelihood combined
	"""
	lp = np.array(lnprior(pars))
	if not np.isfinite(lp):
		return -np.inf, -np.inf
	ll = np.array(lnlike(pars, data, cov))
	return lp + ll, ll

def run_mcmc():
	"""
	Runs the MCMCM, calculating the best match between a model and a dataset based on varying parameters
	"""
	#p0 should be close to the actual values- starting value, they should be spread around the prior 
	p0 = [np.concatenate(((np.random.normal(10, 1,1)), (np.random.normal(12, 1, 1)), (np.random.normal(38, 1,1))))) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (data, cov), threads = 15)
	print('Burn in...')
	pos0, prob, state, blobs = sampler.run_mcmc(p0, burnin) 
	sampler.reset()
	print('Burn in done.')
	n = 1000
	for i, result in enumerate(sampler.sample(pos0, iterations=nsteps)):
		if ((i<1)): t0 = time.time()
		if ((i>0) & (i%n==0)):
			print("{0:5.4%}".format(float(i) / nsteps))
			print('Time remaining: %0.2f min'%((time.time()-t0/float(i)*float(nsteps)-float(i))/60.))
	chain = sampler.flatchain.copy()
	flat_lnlike_samps = sampler.get_blobs(flat=True)
	np.savetxt('chain_red', chain, header='Chain')
	print("Mean acceptance fraction: {0:%.3f}"%(np.mean(sampler.acceptance_fraction)))
	np.savetxt('chain_red_likelihood', flat_lnlike_samps, header='Likelihood')
	return chain
	# The result is a list (of length iterations) of lists (of length nwalkers) of arbitrary objects
	# Note: this will actually be an empty list if your lnpostfn doesn't return any metadata



#Calling the functions
data = xi_red
cov = cov_red
burnin = 100
nsteps = 10000
chain = run_mcmc()
