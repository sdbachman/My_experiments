import numpy
import sys
import re
from netCDF4 import Dataset

def ndgrid(*args, **kwargs):
  same_dtype = kwargs.get("same_dtype", True)
  V = [numpy.array(v) for v in args] # ensure all input vectors are arrays
  shape = [len(v) for v in args] # common shape of the outputs
  result = []
  for i, v in enumerate(V):
    # reshape v so it can broadcast to the common shape
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    zero = numpy.zeros(shape, dtype=v.dtype)
    thisshape = numpy.ones_like(shape)
    thisshape[i] = shape[i]
    result.append(zero + v.reshape(thisshape))
  if same_dtype:
    return numpy.array(result) # converts to a common dtype
  else:
    return result         # keeps separate dtype for each output

def gcd(m,n):
  while n:
     m,n=n,m%n
  return m

def init_tracers(x,y,z,H, L_y, L_x, nx, ny, nz):
  # The odd numbered tracers have a vertical gradient...
  # should we initialize the odd tracers to match the vert/hor gradients?
  # The even numbered tracers have a horizontal gradient
  T1 = numpy.sin(-(numpy.pi * z) / H)
  T3 = numpy.sin(-(2 * numpy.pi * z) / H)
  T5 = numpy.sin(-(3 * numpy.pi * z) / H)
#  T7 = numpy.sin(-(4 * numpy.pi * z) / H)
#  T9 = numpy.sin(-(5 * numpy.pi * z) / H)
#  T11 = numpy.sin(-(6 * numpy.pi * z) / H)
#  T13 = numpy.sin(-(7 * numpy.pi * z) / H)
#  T15 = numpy.sin(-(8 * numpy.pi * z) / H)
#  T17 = numpy.sin(-(9 * numpy.pi * z) / H)
#  T19 = numpy.sin(-(10 * numpy.pi * z) / H)
#  T21 = numpy.sin(-(11 * numpy.pi * z) / H)

  T2 = numpy.sin(numpy.pi * y / L_y)
  T4 = numpy.sin(2 * numpy.pi * y / L_y)
  T6 = numpy.sin(3 * numpy.pi * y / L_y)
#  T8 = numpy.sin(4 * numpy.pi * y / L_y)
#  T10 = numpy.sin(5 * numpy.pi * y / L_y)
#  T12 = numpy.sin(6 * numpy.pi * y / L_y)
#  T14 = numpy.sin(7 * numpy.pi * y / L_y)
#  T16 = numpy.sin(8 * numpy.pi * y / L_y)
#  T18 = numpy.sin(9 * numpy.pi * y / L_y)
#  T20 = numpy.sin(10 * numpy.pi * y / L_y)
#  T22 = numpy.sin(11 * numpy.pi * y / L_y)
  T1 = T1.byteswap()
  T2 = T2.byteswap()
  T3 = T3.byteswap()
  T4 = T4.byteswap()
  T5 = T5.byteswap()
  T6 = T6.byteswap()

  numpy.ravel(T1, 'F').tofile('ptracer/tracers1.bin')
  numpy.ravel(T3, 'F').tofile('ptracer/tracers3.bin')
  numpy.ravel(T5, 'F').tofile('ptracer/tracers5.bin')
#  numpy.ravel(T7, 'F').tofile('ptracer/tracers7.bin')
#  numpy.ravel(T9, 'F').tofile('ptracer/tracers9.bin')
#  numpy.ravel(T11, 'F').tofile('ptracer/tracers11.bin')
#  numpy.ravel(T13, 'F').tofile('ptracer/tracers13.bin')
#  numpy.ravel(T15, 'F').tofile('ptracer/tracers15.bin')
#  numpy.ravel(T17, 'F').tofile('ptracer/tracers17.bin')
#  numpy.ravel(T19, 'F').tofile('ptracer/tracers19.bin')
#  numpy.ravel(T21, 'F').tofile('ptracer/tracers21.bin')

  numpy.ravel(T2, 'F').tofile('ptracer/tracers2.bin')
  numpy.ravel(T4, 'F').tofile('ptracer/tracers4.bin')
  numpy.ravel(T6, 'F').tofile('ptracer/tracers6.bin')
#  numpy.ravel(T8, 'F').tofile('ptracer/tracers8.bin')
#  numpy.ravel(T10, 'F').tofile('ptracer/tracers10.bin')
#  numpy.ravel(T12, 'F').tofile('ptracer/tracers12.bin')
#  numpy.ravel(T14, 'F').tofile('ptracer/tracers14.bin')
#  numpy.ravel(T16, 'F').tofile('ptracer/tracers16.bin')
#  numpy.ravel(T18, 'F').tofile('ptracer/tracers18.bin')
#  numpy.ravel(T20, 'F').tofile('ptracer/tracers20.bin')
#  numpy.ravel(T22, 'F').tofile('ptracer/tracers22.bin')

#---------------------

# genit.m
# This function generates initial condition files for the mitgcm.
# if init_vel=1, then the front problem is used
# if init_vel=0, then resting (other than small symmetry-breaking
# random noise) initial conditions are used.

Ri=10

nx=100 
ny=50 
nz=40

#-- Params
g=9.81                         # GRAVITY
tAlpha=-2.0e-4                 # THERMAL EXPANSION COEFFICIENT
f0=1e-4
rho=1035.0                     # REFERENCE DENSITY
 
H=200.0



N2 = Ri*(f0**2)
M2 = numpy.sqrt(N2 * f0 * f0 / Ri)

N = numpy.sqrt(N2)

# Eddy-resolving: set dx = deformation radius / 5.0
dxspacing = N*H / numpy.abs(f0) / 5.0
dyspacing = dxspacing

print 'dx = ', dxspacing

Lx = dxspacing * nx
Ly = dyspacing * ny

#f = open('data')
#text = f.read()
#f.close()

#SF = re.compile('dxSpacing.*')
#new = SF.sub('dxSpacing=' + str(dxspacing),text)   ##########################################

#f = open('data', 'w')
#f.write(new)
#f.close()

#f = open('data')
#text = f.read()
#f.close()

#SF = re.compile('dySpacing.*')
#new = SF.sub('dySpacing=' + str(dxspacing),text)   ##########################################

#f = open('data', 'w')
#f.write(new)
#f.close()

#f = open('data')
#text = f.read()
#f.close()

#SF = re.compile('delR.*')
#new = SF.sub('delR=40*' + str(H/nz),text)   ##########################################

#f = open('data', 'w')
#f.write(new)
#f.close()


#%%%%%%%%%%%%%%%% GRID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-- Grid: x
dx_ratio=30.0
dx_trans=.05
dx_min=100.0
xn=numpy.arange(0.5, nx+0.5)/nx
dx=numpy.ones(nx)                                  #  uniform resolution
dx=dx*Lx/numpy.sum(dx) 
xf=numpy.cumsum(numpy.insert(dx,[0], 0))           #  Face x points
xc=(xf[0:-1]+xf[1:])/2.0                             #  Centered x points

#-- Grid: y
dy_ratio=30.0
dy_trans=.02
dy_min=100.0
yn=numpy.arange(0.5,ny+0.5)/ny
dy=numpy.ones(ny)                                  # uniform resolution
dy=dy*Ly/numpy.sum(dy) 
yf=numpy.cumsum(numpy.insert(dy, [0], 0))          # Face y-points
yc=(yf[0:-1]+yf[1:])/2;                      # Centered y-points
L=yc[-1]-yc[0]	   # this takes into account the wall of topography!!!

dh=H/nz*numpy.ones(nz)

zf=-numpy.round(numpy.cumsum(numpy.insert(dh, [0],0)))  #face z-points
dh=-numpy.diff(zf)
zc=(zf[0:-1]+zf[1:])/2.0  # centered z points
nz=numpy.alen(dh)

[ZT,YT,XT]=ndgrid(zc,yc,xc) # This is the centered, temperature grid.
[ZU,YU,XU]=ndgrid(zc,yc,xf) # This is the U grid.
[ZV,YV,XV]=ndgrid(zc,yf,xc) # This is the V grid.
[ZW,YW,XW]=ndgrid(zf,yc,xc) # This is the W grid.
[YB,XB]=ndgrid(yc,xc) # This is the Bathymetry grid.

############ Temperature gradients
dTdy = M2 / g / tAlpha
T0=17
dTdz = -N2 / g / tAlpha

#%%%%%%%%% Initial Temp Profile %%%%%%%%%

theta = numpy.zeros((nz,ny,nx))

i=0
iml=nz

while (i<iml):
  theta[i,:,:]=T0+dTdz*(ZT[i,:,:]+H)
  i=i+1

theta = theta+dTdy*YT

#%%%%%%%%%% Thermal wind %%%%%%%%%%%%%%%%%

dens=theta*rho*tAlpha+rho

U = 0.0*XU
print U.shape
#print YU.shape
#k = 1
#print (numpy.diff(dens[k,:,:],1,1)/numpy.diff(YU[k+1,:,:],1,1)).shape
  
for k in range(nz-2,-1,-1):
  U[k,0:-1,0:-1] = U[k+1,0:-1,0:-1]+(dh[k]*g/rho/f0) * numpy.diff(dens[k,:,:],1,0)/numpy.diff(YU[k+1,:,0:-1],1,0)
  U[k,-1,:] = -U[k,-3,:]+2*U[k,-2,:]      # Get endpoint
  U[k,:,-1] = U[k,:,0]            # In the wall

 
##########  CREATE TRACERS #########

#init_tracers(XT, YT, ZT, H, Ly, Lx, nx, ny, nz)

#%%%%%%%% Bathymetry %%%%%%%%%%%%%%%%%%%%
# Bathymetry is on Xc, Yc
#hh=numpy.ones(XB.shape)
#hh[:,-1]=0.0*hh[:,-1]
#hh=-H*hh

#hh = hh.byteswap()
#numpy.ravel(hh, 'F').tofile('topo_sl.bin')

#%%%%%%%%% Make initial state Momentumless (at least boussinesq, good approx)

imax,jmax,kmax=U.shape

#for i in range(0,imax):
#  for j in range(0,jmax):
#    U[:,i,j]=U[:,i,j]-numpy.mean(U[:,i,j])
 

############# Calculate Delta_T ######################

Umax= 20.0 * numpy.amax([numpy.amax(numpy.abs(U[:])), \
numpy.sqrt(numpy.max(N2)*H**2), numpy.sqrt(H**2*M2**2/f0**2)])

print 'Umax =', Umax

deltaT=numpy.round(dxspacing/Umax/30.0)*30.0

print 'Recommended dt = ', deltaT

#Eady_ts = 3.2863 * numpy.sqrt(Ri) / f0
#print 'Eady time scale = ' + str(Eady_ts) + ' s'


#f = open('data')
#text = f.read()
#f.close()

#SF = re.compile('deltaT.*')
#new = SF.sub('deltaT=' + str(deltaT),text)   ##########################################

#SF = re.compile('f0.*')
#new = SF.sub('f0 =' + str(f0),new)

#f = open('data', 'w')
#f.write(new)
#f.close()


# Set timesteps

#f = open('data')
#text = f.read()
#f.close()

#nt = numpy.round(Eady_ts * 20 / deltaT)
#SF = re.compile('nTimeSteps.*')
#new = SF.sub('nTimeSteps='+str(int(nt)),text)   ##########################################

#f = open('data', 'w')
#f.write(new)
#f.close()



#SNAPSHOT_FREQ = numpy.round(nt*deltaT/60.0)  #deltaT*500.0 ##############################################################

# Write stuff to file
#f = open('data.ptracers')
#text = f.read()
#f.close()

#SF = re.compile('PTRACERS_dumpFreq.*')
#new = SF.sub('PTRACERS_dumpFreq=' + str(SNAPSHOT_FREQ), text)

#f = open('data.ptracers', 'w')
#f.write(new)
#f.close()

#f = open('data.diagnostics')
#text = f.read()
#f.close()

#SF1 = re.compile('frequency\(1\).*')
#SF2 = re.compile('frequency\(2\).*')
#SF3 = re.compile('frequency\(3\).*')
#SF4 = re.compile('frequency\(4\).*')
#SF5 = re.compile('frequency\(5\).*')
#SF6 = re.compile('frequency\(6\).*')
#SF7 = re.compile('frequency\(7\).*')
#SF8 = re.compile('frequency\(8\).*')
#SF9 = re.compile('frequency\(9\).*')
#SF10 = re.compile('frequency\(10\).*')
#new = SF1.sub('frequency(1) = ' + str(SNAPSHOT_FREQ), text)
#new = SF2.sub('frequency(2) = ' + str(SNAPSHOT_FREQ), new)
#new = SF3.sub('frequency(3) = ' + str(SNAPSHOT_FREQ), new)
#new = SF4.sub('frequency(4) = ' + str(SNAPSHOT_FREQ), new)
#new = SF5.sub('frequency(5) = ' + str(SNAPSHOT_FREQ), new)
#new = SF6.sub('frequency(6) = ' + str(SNAPSHOT_FREQ), new)
#new = SF7.sub('frequency(7) = ' + str(SNAPSHOT_FREQ), new)
#new = SF8.sub('frequency(8) = ' + str(SNAPSHOT_FREQ), new)
#new = SF9.sub('frequency(9) = ' + str(SNAPSHOT_FREQ), new)
#new = SF10.sub('frequency(10) = ' + str(SNAPSHOT_FREQ), new)

#f = open('data.diagnostics', 'w')
#f.write(new)
#f.close()

#numpy.savetxt('deltaT.txt', [deltaT])
#numpy.savetxt('dxspacing.txt', [dxspacing])
#numpy.savetxt('dzspacing.txt', [H/nz])
#numpy.savetxt('H.txt', [H])
#numpy.savetxt('f0.txt', [f0])
#numpy.savetxt('Ri.txt', [Ri])

#f = open('data')
#text = f.read()
#f.close()

#SF = re.compile('pchkptFreq.*')
#new = SF.sub('pchkptFreq=' + str(SNAPSHOT_FREQ), text)

#f = open('data', 'w')
#f.write(new)
#f.close()


# Add perturbation
pert=numpy.random.rand(ny,nx)

pert=1e-6*(pert-0.5)

for i in range(0,numpy.alen(theta[:,0,0])):
  theta[i,:,:]=theta[i,:,:]+pert


#theta = theta.byteswap()
numpy.ravel(theta, 'F').tofile('thetaInitial.bin')

U2=(U[:,:,0:-1]+U[:,:,1:])/2.0
#U2 = U2.byteswap()
numpy.ravel(U2, 'F').tofile('uInitial.bin')

print U2.shape
thetap = theta
U2p = U2
SALT = numpy.zeros(thetap.shape)
v = numpy.zeros(thetap.shape)

print 'File genit.py completed.'

# Writing NetCDF files
# For this example, we will create two NetCDF4 files. One with the global air
# temperature departure from its value at Darwin, Australia. The other with
# the temperature profile for the entire year at Darwin.
theta_desc = {'name': 'Potential temperature'}

# Simple example: temperature profile for the entire year at Darwin.
# 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
w_nc_fid = Dataset('InitialConditions.nc', 'w', format='NETCDF4')
#w_nc_fid.description = "Initial temperature field" %\
#                      (nc_fid.variables['theta'].var_desc.lower(),\
#                       theta_desc['name'], nc_fid.description)

# Using our previous dimension info, we can create the new time dimension
# Even though we know the size, we are going to set the size to unknown
w_nc_fid.createDimension('x', nx)
w_nc_fid.createDimension('y', ny)
w_nc_fid.createDimension('layers', nz)
w_nc_dim = w_nc_fid.createVariable('PTEMP', 'double',\
                                   ('layers','y','x'))
w_nc_dim = w_nc_fid.createVariable('u', 'double',\
                                   ('layers','y','x'))
w_nc_dim = w_nc_fid.createVariable('SALT', 'double',\
                                   ('layers','y','x'))
w_nc_dim = w_nc_fid.createVariable('v', 'double',\
                                   ('layers','y','x'))
# You can do this step yourself but someone else did the work for us.
#for ncattr in nc_fid.variables['time'].ncattrs():
#    w_nc_dim.setncattr(ncattr, nc_fid.variables['time'].getncattr(ncattr))
# Assign the dimension data to the new NetCDF file.
#w_nc_fid.variables['x'][:] = nx
#w_nc_fid.variables['y'][:] = ny
#w_nc_fid.variables['layers'][:] = nz
#w_nc_var = w_nc_fid.createVariable('air', 'f8', ('time'))
#w_nc_var.setncatts({'long_name': u"mean Daily Air temperature",\
#                    'units': u"degK", 'level_desc': u'Surface',\
#                    'var_desc': u"Air temperature",\
#                    'statistic': u'Mean\nM'})
w_nc_fid.variables['PTEMP'][:] = thetap
w_nc_fid.variables['u'][:] = U2p
w_nc_fid.variables['SALT'][:] = SALT
w_nc_fid.variables['v'][:] = v
w_nc_fid.close()  # close the new file



## Complex example: global temperature departure from its value at Darwin
#departure = air[:, :, :] - air[:, lat_idx, lon_idx].reshape((time.shape[0],\
#                                                             1, 1))
#
## Open a new NetCDF file to write the data to. For format, you can choose from
## 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
#w_nc_fid = Dataset('air.departure.sig995.2012.nc', 'w', format='NETCDF4')
#w_nc_fid.description = "The departure of the NCEP/NCAR Reanalysis " +\
#                      "%s from its value at %s. %s" %\
#                      (nc_fid.variables['air'].var_desc.lower(),\
#                       darwin['name'], nc_fid.description)
## Using our previous dimension information, we can create the new dimensions
#data = {}
#for dim in nc_dims:
#    w_nc_fid.createDimension(dim, nc_fid.variables[dim].size)
#    data[dim] = w_nc_fid.createVariable(dim, nc_fid.variables[dim].dtype,\
#                                        (dim,))
#    # You can do this step yourself but someone else did the work for us.
#    for ncattr in nc_fid.variables[dim].ncattrs():
#        data[dim].setncattr(ncattr, nc_fid.variables[dim].getncattr(ncattr))
## Assign the dimension data to the new NetCDF file.
#w_nc_fid.variables['time'][:] = time
#w_nc_fid.variables['lat'][:] = lats
#w_nc_fid.variables['lon'][:] = lons
#
## Ok, time to create our departure variable
#w_nc_var = w_nc_fid.createVariable('air_dep', 'f8', ('time', 'lat', 'lon'))
#w_nc_var.setncatts({'long_name': u"mean Daily Air temperature departure",\
#                    'units': u"degK", 'level_desc': u'Surface',\
#                    'var_desc': u"Air temperature departure",\
#                    'statistic': u'Mean\nM'})
#w_nc_fid.variables['air_dep'][:] = departure
#w_nc_fid.close()  # close the new file
