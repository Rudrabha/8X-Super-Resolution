imagefiles = dir('Test_Data/*.png');
a_bicdir ='TD/';
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = strcat('Test_Data/',imagefiles(i).name);
   I = imread(currentfilename);
   a = rgb2ycbcr(I);
   b = a(:,:,1);
   imwrite(b,[a_bicdir,imagefiles(i).name]);
end
