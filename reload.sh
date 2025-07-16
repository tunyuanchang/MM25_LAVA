# re run
START=5
EPOCH=5

M0=single
M1=concat
M2=lstm
M3=gru
M4=tcn

echo "Start time: $(TZ=UTC-8 date '+%Y-%m-%d %H:%M:%S')"

# single frame
python3 train.py --epoch  "$EPOCH" \
                 --model "$M0" \
                 --gpu_index 0 \
                 --checkpoint checkpoint_"$M0"_"$START".ckpt
                 > ./result/"$M0"_"$EPOCH".log 2>&1

# concat
python3 train.py --epoch  "$EPOCH" \
                 --model "$M1" \
                 --gpu_index 2 \
                 --checkpoint checkpoint_"$M1"_"$START".ckpt \
                 > ./result/"$M1"_"$EPOCH".log 2>&1 & 

# lstm 
python3 train.py --epoch  "$EPOCH" \
                 --model "$M2" \
                 --gpu_index 3 \
                 --checkpoint checkpoint_"$M2"_"$START".ckpt \
                 > ./result/"$M2"_"$EPOCH".log 2>&1

wait
# gru
python3 train.py --epoch  "$EPOCH" \
                 --model "$M3" \
                 --gpu_index 2 \
                 --checkpoint checkpoint_"$M3"_"$START".ckpt \
                 > ./result/"$M3"_"$EPOCH".log 2>&1 &

# tcn
python3 train.py --epoch  "$EPOCH" \
                 --model "$M4" \
                 --gpu_index 3 \
                 --checkpoint checkpoint_"$M4"_"$START".ckpt \
                 > ./result/"$M4"_"$EPOCH".log 2>&1
wait

echo "End time: $(TZ=UTC-8 date '+%Y-%m-%d %H:%M:%S')"