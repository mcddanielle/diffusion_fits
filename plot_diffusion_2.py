#Danielle McDermott
#July 5, 2017
#Python 2.X

#PROJECT: Granular Bidisperse System
#
#source for keeping xticks while turning off numeric labels:
#http://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from matplotlib.ticker import AutoMinorLocator
#from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
#ticksLocator   = MaxNLocator()
formatter = ticker.ScalarFormatter()
formatter.set_powerlimits((-3,4))
formatter.set_useOffset(False)
formatter.set_scientific(True)
    
import data_importerDM as di
import os, sys

from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * x**b + c

#work on colors,
#make them match at least a little
#may also need to consider transparency and linestyle
#make them a gradient, so when you have many colors, 
#you can visually follow the trend.  
#Light to dark or some such
#have to do it in line with data

#colors=['blue','purple','gray']

'''
N1=10
colors = []
color_idx = np.linspace(0,1,N1)
for i in color_idx:
    colors.append(plt.cm.jet(i))


#this establishes sequential choices of the above colors
#for the plotting.  
#one issue - if you plot more lines than the array length, 
#the code will die
if colors:
    plt.rc('axes',color_cycle=colors)
'''

plt.rc('font',size=18)

############################################################
############################################################
############################################################

def check_array_lengths(array1,array2,array_str="???"):
    
    if len(array1) != len(array2):
        print "Aack! arrays don't match lengths!"
        print array_str
        print len(array1), len(array2)
        exit(-1);
    return


#######################################
#populate a list for annotations
#######################################
def letter_range(start, stop):
    for c in xrange(ord(start), ord(stop)):
        yield chr(c)
    return

letter = []
for x in letter_range('a', 'z'):
    letter.append( x )


################################################
def add_axes(fig1,G1,rows1,columns1):
    '''
    fig1: previously created matplotlib object to put all subplots
    G1:   previously created grid (could do it here, I think)
    rows1:number of rows in grid
    columns1: number of columns in grid
    '''

    for j in range(rows1):
        for i in range(columns1):
            if columns1 > 1 and rows1 > 1:
                ax1 = fig1.add_subplot(G1[j,i])       
                print j,i
            elif columns1 > 1:
                ax1 = fig1.add_subplot(G1[i])       
                print i
            elif rows1 > 1:
                ax1 = fig1.add_subplot(G1[j])       
                #if j>0:
                print j
            else:
                ax1 = fig1.add_subplot(G1[0]) 

    return fig1.get_axes()

###################################################################
#Main Program
###################################################################
#Has been hardcoded to plot some very particular systems

#plot 1 = Avg Big Cluster Size/Num Particles = CL
#plot 2 = <Vx> for interacting/noninteracting
if __name__ == "__main__":
    

    ##################################################################
    A = 8.0; B = 5.0
    columns = 1
    rows = 1
    size1=[columns*A,rows*B]

    #create the figure for plotting
    ############################################
    fig = plt.figure(figsize=(size1[0],size1[1]),
                     num=None, 
                     facecolor='w', 
                     edgecolor='k') 

    G = gridspec.GridSpec(rows, columns) 

    all_axes = add_axes(fig,G,rows,columns)
    #####################
    counter=0
    
    #################################################


    xlabels=[r"$\Delta t$"]
    ylabels=[r'$\langle \delta y^{2} \rangle$']

    file1 = "diffusion_velocity_data.csv"
    path1 = os.getcwd()
    
    try:
        data1 = di.get_data(file1,7,sep=',',path1=path1)

    except:
        print "data read didn't work, exiting now"
        sys.exit()

    
    time = data1[0]
    #fD1 = np.insert(fD1,0,0.0)
    
    phi = 0.01

    dx1 = data1[1]
    dy1 = data1[2]
    dr1 = data1[3]
    
    dx2 = data1[4]
    dy2 = data1[5]
    dr2 = data1[6]

    
    #############################################
    #fit data (fd), take last 10^6 timesteps
    ############################################
    '''
    fit_time = int(1E3)
    time_fd = time[fit_time:]
    dy1_fd = dy1[fit_time:]
    dy2_fd = dy2[fit_time:]
    dx1_fd = dx1[fit_time:]
    dx2_fd = dx2[fit_time:]
    '''
    
    #popt: optimal fit values
    #pcov: covariance of optimal fit values
    
    #if the transverse diffusion is zero, don't fit
    if dy1[-1]>1.0 and np.count_nonzero(dy1) > 0:
        try:
            popty1, pcovy1 = curve_fit(func, time, dy1)
            perr_y1 = np.sqrt(np.diag(pcovy1))
        except:
            print "Fit didn't work"
            popty1 = [0.0,0.0,0.0]
            perr_y1 = [0.0,0.0,0.0]

    else:
        popty1 = [0.0,0.0,0.0]
        perr_y1 = [0.0,0.0,0.0]
        
    #if the transverse diffusion is zero, don't fit
    if dy2[-1]>1.0 and np.count_nonzero(dy2) > 0:
        try:
            popty2, pcovy2 = curve_fit(func, time, dy2)
            perr_y2 = np.sqrt(np.diag(pcovy2))
        except:
            print "Fit didn't work"
            popty2 = [0.0,0.0,0.0]
            perr_y2 = [0.0,0.0,0.0]

    else:
        popty2 = [0.0,0.0,0.0]
        perr_y2 = [0.0,0.0,0.0]

    if np.count_nonzero(dx1) > 0:
        try:
            poptx1, pcovx1 = curve_fit(func, time, dx1)
            perr_x1 = np.sqrt(np.diag(pcovx1))
        except:
            print "Fit didn't work"
            poptx1 = [0.0,0.0,0.0]
            perr_x1 = [0.0,0.0,0.0]

    else:
        poptx1 = [0.0,0.0,0.0]
        perr_x1 = [0.0,0.0,0.0]
        
    if np.count_nonzero(dx2) > 0:
        try:
            poptx2, pcovx2 = curve_fit(func, time, dx2)
            perr_x2 = np.sqrt(np.diag(pcovx2))
        except:
            print "Fit didn't work"
            poptx2 = [0.0,0.0,0.0]
            perr_x2 = [0.0,0.0,0.0]

    else:
        poptx2 = [0.0,0.0,0.0]
        perr_x2 = [0.0,0.0,0.0]

    filename="diffusion_coefficients.txt"
    f = open(filename, 'w') 

    #print pcovy1 #, popty1[1]
    #sys.exit()
    
    #f.write("#alpha1_y covariance(alpha1_y)  alpha1_x cov(a1x) \n")
    f.write("%f %f %f %f "%( popty1[1], 
                               popty2[1], 
                               poptx1[1], 
                               poptx2[1]))
    
    f.write("%f %f %f %f "%( perr_y1[1],
                               perr_y2[1],
                               perr_x1[1],
                               perr_x2[1]))

    f.write("%f %f %f %f \n"%( dy1[-1], 
                               dy2[-1], 
                               dx1[-1], 
                               dx2[-1])) 
    
    f.close()
    
    
    #############################################
    #plotting...
    ############################################
    ax1 = all_axes[0]


    ax1.plot(time,dy1,'-',color='tomato',label="$\delta y^2_1$")
    ax1.plot(time,dy2,'-',color='cornflowerblue',label="$\delta y^2_2$")

    if np.count_nonzero(popty1) > 0:
        ax1.plot(time, func(time, *popty1), '--',color='red',
                 label=r"$\alpha_1=%1.4f \pm %1.4f$"%(popty1[1],perr_y1[1]))

    if np.count_nonzero(popty2) > 0:
        
        ax1.plot(time, func(time, *popty2), '--',color='blue', 
                 label=r"$\alpha_2=%1.4f \pm %1.4f$"%(popty2[1],perr_y2[1]))
    

    ax1.set_ylabel(ylabels[0])
    ax1.set_xlabel(xlabels[0])


    ax1.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax1.xaxis.set_major_formatter(formatter)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax1.legend(loc=2,fontsize=12,ncol=2)

    
    fig.tight_layout()
            
    outname="diffusion_log.png"

    fig.savefig(outname)
    

    sys.exit()


