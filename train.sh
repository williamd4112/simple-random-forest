DEG=10
PRE="pca"
PARAM=""

for n_tree in $1
do
    for min_num in $2
    do
        for frac in $3
        do
            acc=$(python main.py --task train --deg $DEG --pre $PRE --param_rf $n_tree,$min_num,$frac)
            echo $n_tree,$min_num,$frac,$acc
        done
    done
done

