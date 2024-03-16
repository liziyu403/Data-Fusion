%                          wgs84con.m
%  Scope:   This MATLAB macro sets the WGS-84 most used constants as
%           global variables.
%  Usage:   wgs84con
%  Description of global parameters:
%           a_smaxis  -  output, WGS-84 earth semi-major axis in meters
%           b_smaxis  -  output, WGS-84 earth semi-minor axis in meters
%           eccentr   -  output, WGS-84 earth eccentricity 
%           eccentr2  -  output, WGS-84 earth eccentricity squared
%           flatness  -  output, WGS-84 flatness (ellipticity)
%           eprime    -  output, WGS-84 second eccentricity  
%           eprime2   -  output, WGS-84 second eccentricity squared 
%           onemecc2  -  output, WGS-84 one minus eccentricity squared
%           gravpar   -  output, WGS-84 gravity parameter in meters**3/sec**2,
%                        earth's gravitational constant (mass of earth
%                        including earth's atmosphere)
%           rot_rate  -  output, WGS-84 earth rotation rate in radians/second
%           c_speed   -  output, speed of light in vacuum in meters/second
%           ucgrav    -  output, universal constant of gravity (G) in 
%                        meters**3/sec**2/Kg     
%           mearth    -  output, mass of earth (including the atmosphere) (M)
%                        in Kg       
%           g0        -  output, ellipsoidal equatorial gravity in meters/sec**2
%           c20       -  output, normalized second degree zonal harmonic 
%                        coefficient of the gravitational potential (C20)
%  Last update: 06/07/99
%  Copyright (C) 1996-99 by LL Consulting. All Rights Reserved.

global  a_smaxis b_smaxis eccentr eccentr2 flatness eprime eprime2 onemecc2
global  gravpar rot_rate c_speed ucgrav mearth g0 c20

%  Initialize the constants

a_smaxis = 6378137.; 
b_smaxis = 6356752.314245179;
eccentr  = 0.08181919084265;     %  eccentr = sqrt(1 - (b_smaxis/a_smaxis)^2)
eccentr2 = 6.69437999014e-3;     %  eccentr2 = flatness * (2. - flatness)
flatness = 0.00335281066475;     %  1. / 298.257223563 = 1 - b_smaxis/a_smaxis
eprime   = 0.0820944379496;      %  eprime = (a_smaxis/b_smaxis)* eccentr
eprime2  = 6.73949674227e-3;     %  eprime2 = eprime^2
onemecc2 = 0.99330562000986;     %  1. - eccentr2 

gravpar  = 3.986005e+14;
rot_rate = 7.2921151467e-5; 
c_speed  = 2.99792458e+8;
ucgrav   = 6.673e-11;
mearth   = 5.9733328e+24;
g0       = 9.7803327;
c20      = -484.16685e-6;
