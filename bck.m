imagefilesbic = dir('Test_Data/*.png');
imagefiles = dir('Final_RESULTS/*.png');
a_bicdir = 'Final_RESULTS/';
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilenamebic = strcat('Test_Data/',imagefiles(i).name);
   currentfilename = strcat('Final_RESULTS/',imagefiles(i).name);

   I = imread(currentfilename);
   Ibic = imread(currentfilenamebic);
   % figure
   % imshow(I)
    I = rgb2ycbcr(I);
    a = I;
    I = I(:,:,1);
    Ibic = rgb2ycbcr(Ibic);
    Ibic = Ibic(:,:,1);
   finalimg = uint8(backprojection(I, Ibic, 50)); 
   a(:,:,1) = finalimg;
   a = ycbcr2rgb(a);
   %finalimg = finalimg+I;
   imwrite(a,[a_bicdir,imagefiles(i).name]);
end
