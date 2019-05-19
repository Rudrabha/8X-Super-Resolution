imagefiles = dir('Test_Data/*.png');
a_bicdir = 'BIC/';
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = strcat('Test_Data/',imagefiles(i).name);

   I = imread(currentfilename);
   % figure
   % imshow(I)
   I = imresize(I, 8, 'bicubic');
   imwrite(I,[a_bicdir,imagefiles(i).name]);
end
