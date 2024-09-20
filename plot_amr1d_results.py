import numpy as np 
import sys 
import matplotlib.pylab as plt 
MOL=np.arange(10,16,1)
zlist=np.array([10,80])
uxlist=np.array([3,12])
vxlist=np.array([0,0])
for i in range(1,len(sys.argv)):
    data=np.genfromtxt(sys.argv[i])
    z=data[:,0]
    ux=data[:,1]
    uy=data[:,2]
    M=np.sqrt(ux**2+uy**2)
    try:
        temperature=data[:,4]
        tke=data[:,5]
    except:
        temperature=data[:,3]
    plt.figure(1)
    #plt.plot(ux,z,label='U -'+str(MOL[i-1]),lw=4)
    #plt.plot(uy,z,label='V -'+str(MOL[i-1]),lw=4)
    plt.plot(M,z,label=str(MOL[i-1]),lw=4)
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Height [m]')
    plt.ylim(0,2000)
    plt.legend()
    #plt.plot(uy,z,label=str(MOL[i]))
    #plt.plot(uxlist,zlist,'ro')
    plt.legend()
    plt.savefig("hrrr_WindSpeed.png",dpi=600)
    plt.legend()
    plt.figure(2)
    plt.plot(temperature,z,label=str(MOL[i-1]))
    #plt.xlim(300,303)
    plt.legend()
    plt.ylim(0,2000)
    plt.savefig("hrrr_temperature.png",dpi=600)
    # plt.figure(3)
    # plt.plot(lscale,z,label=str(MOL[i-1]))
    # plt.legend()
    # #plt.ylim(0,1000)
    # plt.figure(4)
    # plt.plot(tke,z,'r-o',label=str(MOL[i-1]))
    # plt.legend()
    # #plt.ylim(0,1000)
    # plt.figure(5)
    # plt.plot(nut,z,label=str(MOL[i-1]))
    # plt.legend()
    plt.figure(6)
    M=ux**2+uy**2
    TI=np.sqrt(2.0/3.0*tke)/np.sqrt(M)*100
    plt.plot(TI,z,label=str(MOL[i-1]))
    plt.xlabel('Turbulent Intensity [%]')
    plt.ylabel('Height [m]')
    plt.ylim(0,2000)
    plt.legend()
    plt.savefig("hrrr_TI.png",dpi=600)
    #Rib=9.81*z/temperature[0]*(temperature-temperature[0])/M
    # for i in range(0,len(Rib)):
    #     print(z[i],Rib[i],np.sqrt(2.0/3.0*tke[i])/np.sqrt(ux[i]**2+uy[i]**2))
    #plt.ylim(0,1000)
#for i in range(0,len(z),5):
#    print(z[i])
plt.show()
MOL=[-80]
ug=10
vg=0
if(MOL[0]<0):
    z1=3048
    u1=ug
    v1=vg
    T1=330.48
else:
    z1=2048
    u1=ug
    v1=vg
    T1=320.48
target=open("writeAMR.info","w")
for i in range(0,len(z)):
    if(i==0):
        target.write("ABL.temperature_heights = %g "%(z[i]))
    else:
        target.write(" %g "%(z[i]))
target.write(" %g \n"%(z1))
if(MOL[0]<0):
    z1=3048
    u1=ug
    v1=vg
    T1=330.48
else:
    z1=2048
    u1=ug
    v1=vg
    T1=320.48
    
for i in range(0,len(z)):
    if(i==0):
        target.write("ABL.temperature_values = %g "%(temperature[i]))
    else:
	    target.write(" %g "%(temperature[i]))
target.write("%g \n"%(T1))
for i in range(0,len(z)):
    if(i==0):
        target.write("ABL.u_values = %g "%(ux[i]))
    else:
        target.write(" %g "%(ux[i]))
target.write(" %g \n"%(u1))
for i in range(0,len(z)):
    if(i==0):
        target.write("ABL.v_values = %g "%(uy[i]))
    else:
        target.write(" %g "%(uy[i]))
target.write(" %g \n"%(v1))
target.close()
