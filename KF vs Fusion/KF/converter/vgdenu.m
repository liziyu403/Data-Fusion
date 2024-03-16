%                                vgdenu.m
%  Scope:   This MATLAB macro performs the transformation from geodetic to 
%           ENU (East, North, Up) coordinates for a given position vector
%           specified by the external points in geodetic coordinates; WGS-84 
%           constants are used.
%  Usage:   penu = vgdenu(pgd,pgdref)
%  Description of parameters:
%           pgd    - input, geodetic position of the final point, where
%                    pgd(1) = latitude in radians
%                    pgd(2) = longitude in radians
%                    pgd(3) = altitude in meters
%           pgdref - input, geodetic position of the reference point, where
%                    pgdref(1) = latitude in radians
%                    pgdref(2) = longitude in radians
%                    pgdref(3) = altitude in meters
%           penu  -  output, ENU position vector, with components in meters
%  External Matlab macros used:  tgdecef, vecefenu, wgs84con
%  Last update:  01/07/01
%  Copyright (C) 1996-01 by LL Consulting. All Rights Reserved.

function   penu = vgdenu(pgd,pgdref)

pfinal = tgdecef(pgd(1),pgd(2),pgd(3));
pref = tgdecef(pgdref(1),pgdref(2),pgdref(3));
temp = pfinal - pref;
penu = vecefenu(temp,pgdref(1),pgdref(2));
