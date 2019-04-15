#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import networkx as nx
import os.path
import timeit
import shutil as sh
from progressbar import *
from scipy.sparse import csr_matrix
from scipy.integrate import ode
import argparse



############################# Order Parameter #############################
def r_psi(theta):
    rho = np.mean(np.exp(1j * theta))
    return np.abs(rho), np.angle(rho)
###########################################################################

######################### Complex Kuramoto Model #########################
def sparse_complex(t, phases, N, omega, nu, deltaomega, deltanu, adjext, adjint, t2, sigma, sigma0, beta, beta0, delta, delta0):
    W=phases[:N]
    Z=phases[N:]
    if (t < t2):
        sigmat = sigma0 + (sigma - sigma0) * t / t2
        betat = beta0 + (beta - beta0) * t / t2
        deltat = delta0 + (delta - delta0) * t / t2
    else:
        sigmat = sigma
        betat = beta
        deltat = delta

    return np.concatenate( (1j*W*((omega + deltat*deltaomega + sigmat * np.imag(W.conjugate() * adjext.dot(Z)) + betat * np.imag(W.conjugate() * adjint.dot(Z)))), 1j*Z*((nu + deltat*deltanu + sigmat * np.imag(Z.conjugate() * np.transpose(adjext).dot(W))+betat * np.imag(Z.conjugate() * np.transpose(adjint).dot(W))))) )
###########################################################################

############################# Adiabatic parameter change #############################
def runsim (n, t1, t2, t3, dt, avgcount, thrs, beta0, beta, sigma0, sigma, delta0, delta, seed, seed2, filebase, output):
    start = timeit.default_timer()
    np.random.seed(seed2)

    # Network structure
    N=n
    k=args.k
    numlinks=args.numlinks
    A1=np.zeros((N,N))
    A2=np.zeros((N,N))
    A3=np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(N):
            if(abs((k1-k2)%n) <= k):
                A1[k2,k1]=1
        A1[k1,k1]=0
        A3[k1,k1]=1
    alter=np.random.choice(range(len(np.where((A1+A3)==0)[0])),numlinks,replace=False)
    for alt in alter:
        A2[np.where((A1+A3)==0)[0][alt],np.where((A1+A3)==0)[1][alt]]=np.sign(numlinks)

    adjext = csr_matrix(A1+A2)
    adjint = csr_matrix(A3)


    # Natural frequencies
    omega=np.zeros(N)
    nu=np.zeros(N)
    omega[:]=args.omega1
    nu[:]=args.omega2
    deltaomega= (np.random.random(N)-0.5)
    deltanu= (np.random.random(N)-0.5)

    # Initial phases, from file if it exists
    np.random.seed(seed)
    TETA_init = np.zeros(2*N)

    if os.path.isfile(filebase + 'ic.dat'):
        theta_init = np.loadtxt(filebase + 'ic.dat')
        print('using initial conditions from file')
        sys.stdout.flush()
        TETA_init[:N]=theta_init[0:2*N:2]
        TETA_init[N:]=theta_init[1:2*N:2]

    else:
        TETA_init = 2*np.pi * np.random.random(2*N)
        print('using random initial conditions')
        sys.stdout.flush()

    # Integration
    if t2 == 0:
        t2=dt
    if t3 == 0:
        t3=dt
    pbar=ProgressBar(widgets=['Integration: ', Percentage(),Bar(), ' ', ETA()], maxval=t1)
    pbar2=ProgressBar(widgets=['Clustering: ', Percentage(),Bar(), ' ', ETA()], maxval=int((t1-t3)/dt)-avgcount)
    if output:
        pbar.start()

    phases_init = np.exp(1j * TETA_init)
    rode=ode(sparse_complex).set_integrator('zvode',rtol=0,atol=1e-6,max_step=dt)
    rode.set_initial_value( phases_init, 0 )
    rode.set_f_params(N, omega, nu, deltaomega, deltanu, adjext, adjint, t2, sigma, sigma0, beta, beta0, delta, delta0)
    TETA=np.ndarray( (int(t1/dt), 2*N) )
    dTETA=np.ndarray( (int(t1/dt), 2*N) )

    for n in range(int(t1/dt)):
        t=n*dt
        if output:
            pbar.update(t)
        phases=rode.integrate(rode.t + dt)
        TETA[n] = np.angle(phases)
        dTETA[n] = np.real(sparse_complex(t, phases, N, omega, nu, deltaomega, deltanu, adjext, adjint, t2, sigma, sigma0, beta, beta0, delta, delta0)/(1j*phases))
    if output:
        pbar.finish()

    theta=np.zeros((int(t1/dt),2*N))
    dtheta=np.zeros((int(t1/dt),2*N))
    theta[:, 0:2*N:2] = TETA[:, :N]
    theta[:, 1:2*N:2] = TETA[:, N:]
    dtheta[:, 0:2*N:2] = dTETA[:, :N]
    dtheta[:, 1:2*N:2] = dTETA[:, N:]

    # Order parameter and frequency averaging
    r = np.zeros(int(t1/dt))
    psi = np.zeros(int(t1/dt))
    for j in range(int(t1/dt)):
        r[j], psi[j] = r_psi(theta[j, :])

    #Clustering
    if output:
        pbar2.start()
    # Find oscillators that are locked avgcount steps in a row, and record their mean frequencies
    avgclustertimes=[]
    avglockedtimes=[]
    for i in range(0,int((t1-t3)/dt)-avgcount,avgcount):
        if output:
            pbar2.update(i)
        avgclusters=[]
        avglocked=np.zeros(2*N)+1
        for i2 in range(avgcount):
            locked=np.zeros(2*N)
            argsort=np.argsort(dtheta[int(t3/dt)+i+i2])
            for j in range(1,2*N):
                if abs(dtheta[int(t3/dt)+i+i2,argsort[j]]-dtheta[int(t3/dt)+i+i2,argsort[j-1]])<thrs:
                    locked[argsort[j]]=1
                    locked[argsort[j-1]]=1
            avglocked[np.where(locked==0)[0]]=0
        for j in np.where(avglocked==1)[0]:
            avgclusters.append(np.mean(dtheta[int(t3/dt)+i:int(t3/dt)+i+avgcount-1,j]))
        avgclustertimes.append(avgclusters)
        avglockedtimes.append(avglocked)

    if output:
        pbar2.finish()

    #Find mean number of locked oscillators and number of clusters
    meanlocked=0
    meanclusters=0
    count=0
    for clusters in avgclustertimes:
        meanlocked+=1.0*np.size(clusters)
        if(np.size(clusters)>0):
            numclusters=1.0
            for j in range(1,np.size(clusters)):
                if(abs(np.sort(clusters)[j]-np.sort(clusters)[j-1])>thrs):
                    numclusters+=1.0
        else:
            numclusters=0

        meanclusters+=numclusters
        count+=1
    meanlocked /= count
    meanclusters /= count

    # Output
    f = open(filebase + 'order.dat', 'a+')
    print(sigma, delta, np.mean(r[int(t3 / dt):]), meanlocked, sep=' ', file=f)

    f.close()
    np.savetxt(filebase+'fs.dat', theta[-1, :] % (2 * np.pi))

    if output:
        print(filebase, delta, sigma, np.mean(r[int(t3 / dt):]), meanlocked, meanclusters, np.sum(A2), seed, t2)

        f = open(filebase + 'out.dat', 'w')
        print('./faraday.py %i %f %f %f %f %i %i %f %f %f %f'%(N, t1, t2, t3, dt, avgcount, thrs, beta0, beta, sigma0, sigma), file=f)
        for i in range(N):
            print(omega[i] + delta*deltaomega[i], nu[i] + delta*deltanu[i], sep=' ', file=f, end=' ')
        print('',file=f)
        print(*(r), sep=' ', file=f)
        for i in range(0, 2*N):
            print(*(theta[int(t3 / dt) - 1:, i]), sep=' ', file=f)
        for i in range(0, N):
            print(*(A1[i]+A2[i]+A3[i]), sep=' ', file=f)
        f.close()

        f = open(filebase + 'clusters.dat', 'w')
        for index in range(len(avgclustertimes)):
            print(*(avglockedtimes[index]), file=f, end=' ')
            print(*(avgclustertimes[index]), file=f)
        f.close()

    stop = timeit.default_timer()
    print('runtime: %f' % (stop - start))
    return [sigma, delta, np.mean(r[int(t3 / dt):]), meanlocked]
######################################################################################

############################# Adiabatic coupling branch sweep #############################
def branchsigmasweep(N, t1, t2, t3, dt, avgcount, thrs, sigma0, beta, rthrs, pthrs, delta, dsigma, seed2, filebase):
    start = timeit.default_timer()

    cont=True
    if(os.path.isfile(filebase+"sweep.dat")):
        os.system('rm '+ filebase+"sweep.dat")

    #forward
    sigma=sigma0
    sh.copyfile(filebase+"ic.dat",filebase+"currentic.dat")
    ret=runsim(N, t1, t2, t3, dt, avgcount, thrs, beta, beta, np.round(sigma,3), np.round(sigma,3), np.round(delta,4), np.round(delta,4), 1, seed2, filebase+"current", 0)

    while cont:
        print('forward ', *ret)
        sh.copyfile(filebase+"currentfs.dat",filebase+"currentic.dat")
        ret=runsim(N, t1, t2, t3, dt, avgcount, thrs, beta, beta, np.round(sigma,3), np.round(sigma+dsigma,3), np.round(delta,4), np.round(delta,4), 1, seed2, filebase+"current", 0)
        sigma=sigma+dsigma

        outs=np.loadtxt(filebase+"currentorder.dat")
        diffs=np.diff(outs,axis=0)[-1]
        if np.abs(diffs[2]) > rthrs or np.abs(diffs[3])>pthrs or np.round(sigma,3)>=0.6 or np.round(sigma,3)<=0.2:
            cont=False
    print('forward end ', *ret)
    sigma2=sigma-dsigma

    if(os.path.isfile(filebase+"currentorder.dat")):
        os.system('sed \$d ' + filebase+"currentorder.dat >> " + filebase + "sweep.dat")
        os.remove(filebase+"currentorder.dat")
        os.remove(filebase+"currentic.dat")
        os.remove(filebase+"currentfs.dat")

    #backward
    sigma=sigma0
    cont=True
    sh.copyfile(filebase+"ic.dat",filebase+"currentic.dat")
    ret=runsim(N, t1, t2, t3, dt, avgcount, thrs, beta, beta, np.round(sigma,3), np.round(sigma,3), np.round(delta,4), np.round(delta,4), 1, seed2, filebase+"current", 0)

    while cont:
        print('backward ', *ret)
        sh.copyfile(filebase+"currentfs.dat",filebase+"currentic.dat")
        ret=runsim(N, t1, t2, t3, dt, avgcount, thrs, beta, beta, np.round(sigma,3), np.round(sigma-dsigma,3), np.round(delta,4), np.round(delta,4), 1, seed2, filebase+"current", 0)
        sigma=sigma-dsigma

        outs=np.loadtxt(filebase+"currentorder.dat")
        diffs=np.diff(outs,axis=0)[-1]
        if np.abs(diffs[2]) > rthrs or np.abs(diffs[3])>pthrs or np.round(sigma-dsigma,3)>0.6 or np.round(sigma+dsigma,3)<0.2:
            cont=False

    print('backward end ', *ret)
    sigma1=sigma+dsigma
    crits=[sigma1, sigma2]
    np.savetxt(filebase+'crits.dat',crits, fmt="%.12f")

    if(os.path.isfile(filebase+"currentorder.dat")):
        os.system('sed \$d ' + filebase+"currentorder.dat >> " + filebase + "sweep.dat")
        os.remove(filebase+"currentorder.dat")
        os.remove(filebase+"currentic.dat")
        os.remove(filebase+"currentfs.dat")

    stop = timeit.default_timer()
    print('sweeptime: %f' % (stop - start))

###########################################################################################
#Command line arguments
parser = argparse.ArgumentParser(description='Moving mesh simulation for inviscid Faraday waves with inhomogeneous substrate.')
parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output')
parser.add_argument("--sweep", type=bool, required=False, dest='sweep', default=False, help='Flag to run a branch sweep')
parser.add_argument("--output", type=bool, required=False, dest='output', default=True, help='Flag to output data')
parser.add_argument("--number", type=int, required=False, dest='number', default=50, help='Number of Janus oscillators')
parser.add_argument("--time", type=float, required=False, dest='time', default=10000., help='Total integration time')
parser.add_argument("--atime", type=float, required=False, dest='atime', default=0., help='Time to adiabatically change coupling from sigma0 to sigma')
parser.add_argument("--rtime", type=float, required=False, dest='rtime', default=9000., help='Time to start averaging order parameter')
parser.add_argument("--dt", type=float, required=False, dest='dt', default=0.1, help='Time between outputs')
parser.add_argument("--avgcount", type=int, required=False, dest='avgcount', default=10, help='Number of timesteps to average over for cluster counting')
parser.add_argument("--threshold", type=float, required=False, dest='threshold', default=0.01, help='Frequency difference threshold for cluster counting')
parser.add_argument("--beta0", type=float, required=False, dest='beta0', default=0.25, help='Initial internal coupling constant')
parser.add_argument("--beta1", type=float, required=False, dest='beta1', default=0.25, help='Final internal coupling constant')
parser.add_argument("--sigma0", type=float, required=False, dest='sigma0', default=0.4, help='Initial external coupling constant')
parser.add_argument("--sigma1", type=float, required=False, dest='sigma1', default=0.4, help='Final external coupling constant')
parser.add_argument("--delta0", type=float, required=False, dest='delta0', default=0.0, help='Initial frequency heterogeneity')
parser.add_argument("--delta1", type=float, required=False, dest='delta1', default=0.0, help='Final frequency heterogeneity')
parser.add_argument("--omega1", type=float, required=False, dest='omega1', default=0.5, help='Natural frequency 1')
parser.add_argument("--omega2", type=float, required=False, dest='omega2', default=-0.5, help='Natural frequency 2')
parser.add_argument("--iseed", type=int, required=False, dest='iseed', default=5, help='Initial condition seed')
parser.add_argument("--hseed", type=int, required=False, dest='hseed', default=1, help='Heterogeneity profile seed')
parser.add_argument("--dsigma", type=float, required=False, dest='delta1', default=0.002, help='Coupling strength sweep step size')
parser.add_argument("--rthreshold", type=float, required=False, dest='rthres', default=0.05, help='Threshold change in order parameter to stop branch sweep')
parser.add_argument("--pthreshold", type=float, required=False, dest='pthres', default=5.0, help='Threshold change in num locked to stop branch sweep')
parser.add_argument("--neighbors", type=int, required=False, dest='k', default=1, help='Number of neighbors')
parser.add_argument("--links", type=int, required=False, dest='numlinks', default=0, help='Number of neighbors')
args = parser.parse_args()

if not (args.sweep):

    n = args.number  # oscillators
    t1 = args.time  # total time
    t2 = args.atime  # time to adiabatically change sigma
    t3 = args.rtime  # time to start averaging r
    dt = args.dt  # timestep
    avgcount = args.avgcount  # clustering averaging count
    thrs = args.threshold  # clustering threshold
    beta0 = args.beta0  # initial internal coupling strength
    beta = args.beta1  # final internal coupling strength
    sigma0 = args.sigma0 # initial external coupling strength
    sigma = args.sigma1  # final external coupling strength
    delta0 = args.delta0 # initial heterogeneity strength
    delta = args.delta1 # final heterogeneity strength
    seed = args.iseed  # random seed
    seed2 = args.hseed  # random seed
    filebase = args.filebase  # output file name
    output = args.output  # output flag
    runsim (n, t1, t2, t3, dt, avgcount, thrs, beta0, beta, sigma0, sigma, delta0, delta, seed, seed2, filebase, output)

else:
    n = args.number  # oscillators
    t1 = args.time  # total time
    t2 = args.atime  # time to adiabatically change sigma
    t3 = args.rtime  # time to start averaging r
    dt = args.dt  # timestep
    avgcount = args.avgcount  # clustering averaging count
    thrs = args.threshold  # clustering threshold
    sigma0 = args.sigma0 # initial external coupling strength
    beta = args.beta1  # internal coupling strength
    delta = args.delta1 # final heterogeneity strength
    dsigma = args.dsigma # coupling strength step size
    rthrs = args.rthreshold # order parameter threshold change
    pthrs = args.pthreshold # num locked threshold change
    seed2 = args.hseed  # random seed for heterogeneity profile
    filebase = args.filebase  # output file name
    branchsigmasweep(n, t1, t2, t3, dt, avgcount, thrs, sigma0, beta, rthrs, pthrs, delta, dsigma, seed2, filebase)
