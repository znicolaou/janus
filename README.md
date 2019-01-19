# Files in the janus repository 
The file janus.py is a python script to numerically integrate Janus oscillator networks using SciPy's Complex-valued Variable-coefficient Ordinary Differential Equation solver.  The file plot.nb is a Mathematica notebook which can be used to plot results.
# System requirements
This python code has been run with anaconda2, which can be downloaded from the Anaconda website: https://www.anaconda.com/download/#macos.  It requires packages numpy, networkx, scipy, and progressbar, which can be installed with `pip install numpy scipy networkx progressbar` after installing anaconda2.  The Mathematica code has been run with Mathematica 11.1.1.0.
# Usage
Running the terminal command `python janus.py` will produce the following usage message.  
Usage will depend on number of command line argument  
usage 1: python explosive.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [sigma0] [sigma] [delta0] [delta] [seed] [seed2] [output] [filebase]  
Used for single simulation runs with adiabatic parameter changes  
N is the number of oscillators  
t1 is the total integration time  
t2 is the time to adiabatically change coupling from sigma0 to sigma  
t3 is the time to start averaging r  
dt is the time between outputs  
avgcount is the number of timesteps to average over for cluster counting  
thrs is the frequency difference threshold for cluster counting  
beta0 is the initial internal coupling constant  
beta is the final internal coupling constant  
sigma0 is the initial external coupling constant  
sigma is the final external coupling constant  
delta0 is initial frequency heterogeneity   
delta is final frequency heterogeneity   
seed is random seed for initial condition  
seed2 is random seed for heterogeneity  
output is 1 to output time data  
filebase is the output file string base; output files are filebaseout.dat and filebaseorder.dat  
example: python janus.py 50 1000 0.1 0.1 0.1 10 1e-2 0.25 0.25 0.35 0.35 0.0 0.0 1 1 data/test/test 1  

usage 2: python explosive.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [sigma0] [sigma] [delta0] [delta] [dsigma] [rthrs] [pthrs] [seed] [seed2] [filebase]   
Used to adiabatically sweep out the coupling constant of a solution branch.  Stop when the change in mean order parameter of mean phase locked oscillators changes by more than rthrs and pthrs, respectively.
N is the number of oscillators  
t1 is the total integration time  
t2 is the time to adiabatically change coupling from sigma0 to sigma  
t3 is the time to start averaging r  
dt is the time between outputs  
avgcount is the number of timesteps to average over for cluster counting  
thrs is the frequency difference threshold for cluster counting  
beta0 is the initial internal coupling constant  
beta is the final internal coupling constant  
sigma0 is the initial external coupling constant  
sigma is the final external coupling constant  
delta0 is initial frequency heterogeneity   
delta is final frequency heterogeneity   
dsigma is the coupling strength step  
rthrs is the threshold change in order parameter to stop branch sweep  
pthrs is the threshold change in num locked to stop branch sweep  
seed is random seed for initial condition  
seed2 is random seed for heterogeneity  
filebase is the output file string base; the program assumes a filebaseic.dat exists for the initial condition.  
example: python janus.py 50 10000 4000 5000 0.1 10 1e-2 0.55 0.25 0.0 0.004 0.02 5.0 data/syncbranch/sync  
___
# Output files
Usage 1  
The program always appends a summary of statistics about the simulation to the file filebaseorder.dat, which contains the coupling constant sigma, the heterogeneity strength delta, the mean order parameter r, and the mean number of phase-locked oscillators Nlocked, where the means are calculated from time t2, to t1. The program also always saves the final state of the system to filebasefs.dat (which can be copied to another filebaseic.dat to use as initial conditions elsewhere). If the output flag is set to 1, the program also creates a file filebaseout.dat that contains more detailed data.  The first line of filebaseout.dat contains the command line that ran the program.  The second line contains the natural frequencies of each phase oscillator in each Janus oscillator.  The third line contains the instantaneous order parameter at each timestep.  The next 2N lines contains the phases of each phase oscillator in each Janus oscillator for each timestep. The next N lines contains the adjacency matrix of the network of phase oscillators.  

Usage 2  
The program creates a filebasesweep.dat, which contains the summaries in the form of the filebaseorder.dat from usage 1 for each sigma in the branch sweep.  The program also creates a filebasecrits.dat which contains the smallest and largest values of the coupling constant sigma for which the branch exists.
