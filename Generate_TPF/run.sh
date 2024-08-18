# dir_path='/NSCH2D'
cd $dir_path
# generator epsilon from 0.01 to 1,default is 0.02
./generate_eps &
# generator Re from 1 to 100, default is 100
./generate_Re &
# generator mobility from 0.1 to 10,default is 0.1
./generate_mob &
# generator Ca from 1 to 100, default is 1
./generate_Ca &
# generator phi randomly range  -1 to 1
./generate_phi &
# generator initial and boundary condition defined by C0, from 0 to 10, but not include 0
./generate_ibc &

wait

echo "All done"