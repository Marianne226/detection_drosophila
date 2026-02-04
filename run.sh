# sudo mount -t tmpfs -o size=2G tmpfs /mnt/tmpfs/
make
cd pulse/
g++ pulse.cpp -o pulse -lphidget22
taskset -c 7 ./pulse &
pid=$!
cd ../
# datetime=$(date +"%Y%m%d-%H%M%S")
./flir-control 20 | python gui.py
# mkdir -p data
# mv /mnt/tmpfs/temp.mp4 data/${datetime}.mp4
# mv /mnt/tmpfs/temp.bin data/${datetime}.bin
kill $pid
wait $pid
# sudo umount /mnt/tmpfs/