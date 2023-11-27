samples=3
# samples= 2 #3
window_step=10
# window_step= 34 #20
epsilon_100=20
# epsilon=$(echo "scale=6; $epsilon_100 / 1000000" | bc)
# epsilon=$(echo "scale=3; $epsilon_100 / 1" | bc)
epsilon=$(echo "scale=3; $epsilon_100 / 100" | bc)
echo "epsilon",$epsilon
# samples= 2 #3
echo "window step",$window_step
echo "samples", $samples
# echo "/g"

# example.txt should always stay the same
sed -e "s/epsilon = .4/epsilon = $epsilon/g" -e "s/window_step = 17/window_step = $window_step/g" -e "s/window_size = 34/window_size = $window_step/g" -e "s/samples = 2/samples = $samples/g" configs/example.txt > configs/example.ini

python nlmap.py 