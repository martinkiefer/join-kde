python2 plot.py

for dir in ./*/
do  
    echo $dir
    cd $dir
    gnuplot plot.gnuplot > "../${dir%?}.pdf"
    cd ..
done
