for dir in ./*/
do  
    rm -f ${dir}/code/*.group.results # 2> /dev/null
    for rf in ${dir}code/*.results
    do
        echo $rf
        python groupBy.py $rf > $rf.group.results
    done 
done

python2 plot.py

for dir in ./*/
do  
    cd $dir
    gnuplot plot.gnuplot > "../${dir%?}.pdf"
    cd ..
done
