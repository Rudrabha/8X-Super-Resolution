imagefilesbic = dir('BIC/*.png');
imagefiles = dir('Res-2/*.png');
a_bicdir = 'Res-2/';
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = strcat('Res-2/',imagefiles(i).name);
   currentfilenamebic = strcat('BIC/',imagefiles(i).name);

   I = imread(currentfilename);
   Ibic = imread(currentfilenamebic);
   % figure
   % imshow(I)
 
   b=rgb2ycbcr(Ibic);
   b(:,:,1) = I;
   finalimg = ycbcr2rgb(b);
   imwrite(finalimg,[a_bicdir,imagefiles(i).name]);
end
