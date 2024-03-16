%                             tgdecef.m
%  Scope:   This MATLAB macro performs the transformation from geodetic to
%           ECEF coordinates for a given position; WGS-84 constants are used.
%  Usage:   pecef = tgdecef(lat,lon,alt)
%  Description of parameters:
%           lat   -  input, latitude of the given position, in radians
%           lon   -  input, longitude of the given position, in radians
%           alt   -  input, altitude (above ellipsoid) of the given position,
%                    in meters
%           pecef -  output, ECEF position, with components in meters
%  External Matlab macros used:  wgs84con
%  Last update:  04/04/00
%  Copyright (C) 1996-00 by LL Consulting. All Rights Reserved.

function   pecef = tgdecef(lat,lon,alt)

%  Initialize the constants

wgs84con
% global constants used:  a_smaxis, eccentr2, onemecc2

%  Compute ECEF position

slat = sin(lat);
clat = cos(lat);
temp = a_smaxis / sqrt(1. - eccentr2 * slat * slat);
pecef = zeros(3,1);
pecef(1) = (temp + alt) * clat * cos(lon);
pecef(2) = (temp + alt) * clat * sin(lon);
pecef(3) = (alt + temp * onemecc2) * slat;
