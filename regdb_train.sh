for trail in 1 2 3 4 5 6 7 8 9 10
#for trail in 9 10
do
  python train.py --dataset regdb --gpu 6 --trial $trail --method awg
done
echo 'Done!'