
for tol in {.6,.65,.7,.75}
do
    for bb in {15,20}
    do
        rm -r cache/
        mkdir cache
        python nlmap.py -t $tol -n $bb >> out.txt
        echo "tol $tol bb $bb" >> searchresults.txt
        cat cache/gt_stats.txt >> searchresults.txt
    done
done