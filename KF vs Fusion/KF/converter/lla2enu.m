function A = lla2enu(lat,lon,alt,ref)
%tranformation de l'espace lat/long/alt vers le repère local ENU
% lat : vecteur latitude (degré)
% lon : vecteur longitude (degré)
% alt : vecteur altitude (m)
% ref : position référence avec
%       ref(1) : latitude (degré)
%       ref(2) : longitude (degré)
%       ref(3) : altitude (m)
if(length(ref) == 3)
   if(length(lat)==length(lon) && length(lat)==length(alt))
      n= length(lat);
      A=zeros(n,3);
      for i=1:n
          A(i,:) = vgdenu([lat(i)*pi/180.0 lon(i)*pi/180.0 alt(i)],[ref(1)*pi/180.0 ref(2)*pi/180.0 ref(3)]);
      end
   end
end