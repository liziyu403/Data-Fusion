%% load data
clear;clc;
% Load Sat data
% col 1 : time
% col 2 : nbr of sat
% col 3 : PRN of sat
% col 4 5 6 : Pos of sat in ENU
% col 7 : Pseudo range of each satellite
% (3 4 5 6 7) is repeated
sat_Data = dlmread('Source/satellite_data.txt');

%Load pos_data
% col 1 : time
% col 2 3 4: Pos of the receiver in ENU
GPS_Data = load('Source/position_data.txt');
% Load ground truth
% col 1 2 3: position of the ground truth in (Latitude,Longitude, altitude)
% col 4 5 6: position of the ground truth in ENU
% col 7 : time
pos_GT   = load('Source/fich_lla_GT');

ref  = load('Source/reference_ENU');
ref_lla  = [ref(2,1)*pi/180 ref(2,2)*pi/180 ref(2,3)]';
ref_ecef = ref(1,:)';
addpath(genpath('converter'));
bounds = [2.164300000000000   2.196600000000000
  48.701999999999998  48.716500000000003];
bounds_enu = lla2enu(bounds(2,:),bounds(1,:),[ref(2,3) ref(2,3)],ref(2,:));

L_sat_Data = size(sat_Data,1);
C_sat_Data = size(sat_Data,2);
for i=1:L_sat_Data
   Satellite(i).time   = sat_Data(i,1);
   Satellite(i).NbrSat = sat_Data(i,2);
   Satellite(i).PRN    = sat_Data(i,3:5:C_sat_Data);
   Satellite(i).X      = sat_Data(i,4:5:C_sat_Data);
   Satellite(i).Y      = sat_Data(i,5:5:C_sat_Data);
   Satellite(i).Z      = sat_Data(i,6:5:C_sat_Data);
   Satellite(i).PR     = sat_Data(i,7:5:C_sat_Data); 
end