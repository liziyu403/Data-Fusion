%                                vecefenu.m
%  Scope:   This MATLAB macro performs the transformation from ECEF frame to 
%           ENU (East, North, Up) frame for a given position vector and
%           referenced latitude/longitude angles.
%  Usage:   penu = vecefenu(pecef,lat,lon)
%  Description of parameters:
%           pecef -  input, ECEF position vector (with components in meters)
%           lat   -  input, reference latitude in radians
%           lon   -  input, reference longitude in radians
%           penu  -  output, ENU position vector, with components in same 
%                    units as the input ECEF position vector
%  Last update:  04/04/00
%  Copyright (C) 1996-00 by LL Consulting. All Rights Reserved.

function   penu = vecefenu(pecef,lat,lon)

slat = sin(lat);
clat = cos(lat);
slon = sin(lon);
clon = cos(lon);

penu = zeros(3,1);
penu(1) = - slon * pecef(1) + clon * pecef(2);
temp = clon * pecef(1) + slon * pecef(2);
penu(2) = - slat * temp + clat * pecef(3);
penu(3) = clat * temp + slat * pecef(3);