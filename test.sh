rm TD/*.png
rm BIC/*.png
rm Res-1/*.png
rm Res-2/*.png
rm Final_RESULTS/*.png
matlab -nodesktop -nosplash -r "run eightxbic.m; exit();"
matlab -nodesktop -nosplash -r "run rgbtoycbcr.m; exit();"
python createtd.py
python 8XSR_test.py
python createtd_cascade.py
python 8XSR_cascaded_test.py
matlab -nodesktop -nosplash -r "run ycbcrtorgb.m; exit();"
python createtd_3D.py
python 8XSR_3D_test.py
matlab -nodesktop -nosplash -r "run bck.m; exit();"
