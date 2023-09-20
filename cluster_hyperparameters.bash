# for i in {1..1}
# do
    
#     sed 's/learned = False/learned = False/g' configs/example.txt > configs/example.ini
#     # var="hi"
#     # echo $configs
#     # echo $configs > 'configs/example.ini'
#     # echo $configs > 'configs/example.txt'
#     # echo $configs >> 'configs/example.ini'
# #    python nlmap.py
# done

######################### run for vild and pose

# for samples in {2..4..2}
# do
#     for window_step in {1..17..8} # can't do 34 because it'll error when window is 17. make this a smaller range
#     do
#         for epsilon_100 in {10..30..5} # 0.3 - 0.6 with 0.05 increment
#         # $(seq 0.01 5.0 0.01)
#         do
#             epsilon=$(bc <<<"scale=2; $epsilon_100 / 100" )
#             # echo $epsilon
#             # echo $window_step
#             # echo $window_size

#             # example.txt should always stay the same
#             sed -e "s/epsilon = .35/epsilon = $epsilon/g" -e "s/window_step = 17/window_step = $window_step/g" -e "s/window_size = 35/window_size = 34/g" -e "s/samples = 5/samples = $samples/g" configs/example.txt > configs/example.ini
            
#             python nlmap.py
#             python delete_figs_clusters.py
#         done
#     done
# done


######################### run loop for pose

# sed 's/only_pose = False/only_pose = True/g' configs/example.txt > configs/example.ini
# sed 's/vild_and_pose = True/vild_and_pose = False/g' configs/example.txt > configs/example.ini


# for samples in {2..4..2}
# do
#     for window_step in {9..17..8} # can't do 34 because it'll error when window is 17
#     do
#         for epsilon_100 in {10..55..5} # 0.1 - 0.55 with 0.05 increment
#         # $(seq 0.01 5.0 0.01)
#         do
#             epsilon=$(bc <<<"scale=2; $epsilon_100 / 100" )
#             # echo $epsilon
#             # echo $window_step
#             # echo $window_size
            
#             # example.txt should always stay the same
#             sed -e "s/epsilon = .35/epsilon = $epsilon/g" -e "s/window_step = 17/window_step = $window_step/g" -e "s/window_size = 35/window_size = 34/g" -e "s/samples = 5/samples = $samples/g" configs/example.txt > configs/example.ini
            
#             python nlmap.py
#         done
#     done
# done


# only_pose = True
# vild_and_pose = False
# learned = False

######################### run loop for learned

# sed 's/only_pose = True/only_pose = False/g' configs/example.txt > configs/example.ini
# sed 's/learned = False/learned = True/g' configs/example.txt > configs/example.ini


for samples in {2..4..2}
# for samples in {3..5..1}
do
    for window_step in {19..17..8} # make sure window_step is not over window_size
    # for window_step in {9..17..8} # make sure window_step is not over window_size
    do
        for epsilon_100 in {1000..2000..50} # 0.01 to 20 with 0.005 increment 0.001 0.005 0.01 0.015 0.02 0.05 0.1 0.15 0.2
        # for epsilon_100 in {1..25..10} # 0.01 to 20 with 0.005 increment 0.001 0.005 0.01 0.015 0.02 0.05 0.1 0.15 0.2
        # for epsilon_100 in {1000..2000..50} # 10 to 20 with 0.5 increment
        do
            epsilon=$(echo "scale=3; $epsilon_100 / 1000" | bc)
            echo "epsilon",$epsilon
            echo "window step",$window_step
            echo "samples", $samples
            
            # example.txt should always stay the same
            sed -e "s/epsilon = .4/epsilon = $epsilon/g" -e "s/window_step = 17/window_step = $window_step/g" -e "s/window_size = 34/window_size = 10/g" -e "s/samples = 2/samples = $samples/g" configs/example.txt > configs/example.ini
            
            python nlmap.py
        done
    done
done



