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
    A1=np.zeros((N,N))
    A2=np.zeros((N,N))
    A3=np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(N):
            if(abs((k1-k2)%n) <= 1):
                A1[k2,k1]=1
        A1[k1,k1]=0
        A3[k1,k1]=1

    adjext = csr_matrix(A1)
    adjint = csr_matrix(A3)

    # Natural frequencies
    omega=np.zeros(N)
    nu=np.zeros(N)
    omega[:]=0.5
    nu[:]=-0.5
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
        print(*(sys.argv), sep=' ', file=f)
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

if(len(sys.argv) != 18 and len(sys.argv) != 16):
    print(len(sys.argv))
    print('Usage will depend on number of command line argument  ')
    print('usage 1: python explosive.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [beta0] [beta] [sigma0] [sigma] [delta0] [delta] [seed] [seed2] [filebase] [output] ')
    print('Used for single simulation runs with adiabatic parameter changes  ')
    print('N is the number of oscillators  ')
    print('t1 is the total integration time  ')
    print('t2 is the time to adiabatically change coupling from sigma0 to sigma  ')
    print('t3 is the time to start averaging r  ')
    print('dt is the time between outputs  ')
    print('avgcount is the number of timesteps to average over for cluster counting  ')
    print('thrs is the frequency difference threshold for cluster counting  ')
    print('beta0 is the initial internal coupling constant  ')
    print('beta is the final internal coupling constant  ')
    print('sigma0 is the initial external coupling constant  ')
    print('sigma is the final external coupling constant  ')
    print('delta0 is initial frequency heterogeneity  ')
    print('delta is final frequency heterogeneity  ')
    print('seed is random seed for initial condition (if filebaseic.dat does not exist, otherwise initial conditions from file are used)  ')
    print('seed2 is random seed for heterogeneity  ')
    print('filebase is the output file string base; output files are filebaseout.dat and filebaseorder.dat  ')
    print('output is 1 to output time data and 0 for shorter output ')
    print('Example: python janus.py 50 10000 0.1 9000 0.1 10 1e-2 0.25 0.25 0.4 0.4 0.0 0.0 5 1 data/random/random 1  ')
    print('This example will likely produce a chimera state.  The final state data/random/randomfs.dat can be copied to chimeraic.dat to save as an initial condition. Run the following command to copy the file for example 2 below. ' )
    print('mkdir -p data/branches && cp data/random/randomfs.dat data/branches/chimeraic.dat  ' )
    print('\n')
    
    print('usage 2: python explosive.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [sigma0] [beta] [delta] [dsigma] [rthrs] [pthrs] [seed2] [filebase]  ')
    print('Used to adiabatically sweep out a solution branch  ')
    print('N is the number of oscillators  ')
    print('t1 is the total integration time  ')
    print('t2 is the time to adiabatically change coupling from sigma0 to sigma  ')
    print('t3 is the time to start averaging r  ')
    print('dt is the time between outputs  ')
    print('avgcount is the number of timesteps to average over for cluster counting  ')
    print('thrs is the frequency difference threshold for cluster counting  ')
    print('sigma0 is the initial external coupling constant  ')
    print('beta is the internal coupling constant  ')
    print('delta is frequency heterogeneity  ')
    print('dsigma is the coupling strength step  ')
    print('rthrs is the threshold change in order parameter to stop branch sweep  ')
    print('pthrs is the threshold change in num locked to stop branch sweep  ')
    print('seed2 is random seed for the heterogeneity profile  ')
    print('filebase is the output file string base; output files are filebaseout.dat and filebaseorder.dat  ')
    print('Example: python janus.py 50 10000 4000 5000 0.1 10 1e-2 0.4 0.25 0.0 0.002 0.05 5.0 1 data/branches/chimera  ')
    print('This will sweep out the coupling constant for the chimera initial condition.  ' )

    exit()


if(len(sys.argv) == 18):

    n = int(sys.argv[1])  # oscillators
    t1 = float(sys.argv[2])  # total time
    t2 = float(sys.argv[3])  # time to adiabatically change sigma
    t3 = float(sys.argv[4])  # time to start averaging r
    dt = float(sys.argv[5])  # timestep
    avgcount = int(sys.argv[6])  # clustering averaging count
    thrs = float(sys.argv[7])  # clustering threshold
    beta0 = float(sys.argv[8])  # initial internal coupling strength
    beta = float(sys.argv[9])  # final internal coupling strength
    sigma0 = float(sys.argv[10])  # initial external coupling strength
    sigma = float(sys.argv[11])  # final external coupling strength
    delta0 = float(sys.argv[12]) # initial heterogeneity strength
    delta = float(sys.argv[13]) # final heterogeneity strength
    seed = int(sys.argv[14])  # random seed
    seed2 = int(sys.argv[15])  # random seed
    filebase = sys.argv[16]  # output file name
    output = int(sys.argv[17])  # output flag
    runsim (n, t1, t2, t3, dt, avgcount, thrs, beta0, beta, sigma0, sigma, delta0, delta, seed, seed2, filebase, output)

elif(len(sys.argv) == 16):
    
    n = int(sys.argv[1])  # oscillators
    t1 = float(sys.argv[2])  # total time
    t2 = float(sys.argv[3])  # time to adiabatically change sigma
    t3 = float(sys.argv[4])  # time to start averaging r
    dt = float(sys.argv[5])  # timestep
    avgcount = int(sys.argv[6])  # clustering averaging count
    thrs = float(sys.argv[7])  # clustering threshold
    sigma0 = float(sys.argv[8]) # initial external coupling strength
    beta = float(sys.argv[9]) # internal coupling strength
    delta = float(sys.argv[10]) # heterogeneity strength
    dsigma = float(sys.argv[11]) # coupling strength step size
    rthrs = float(sys.argv[12]) # order parameter threshold change
    pthrs = float(sys.argv[13]) # num locked threshold change
    seed2 = int(sys.argv[14])  # random seed for heterogeneity profile
    filebase = sys.argv[15]  # output file name
    branchsigmasweep(n, t1, t2, t3, dt, avgcount, thrs, sigma0, beta, rthrs, pthrs, delta, dsigma, seed2, filebase)
