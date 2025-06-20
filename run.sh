# init
EPOCH=5

M0=single
M1=concat
M2=lstm
M3=gru
M4=tcn

echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

# single frame
python3 train.py --epoch  "$EPOCH" \
                 --model "$M0" \
                 --gpu_index 0 \
                 > ./result/"$M0"_"$EPOCH".log 2>&1

#concat
python3 train.py --epoch  "$EPOCH" \
                 --model "$M1" \
                 --gpu_index 0 \
                 > ./result/"$M1"_"$EPOCH".log 2>&1 & 

#lstm 
python3 train.py --epoch  "$EPOCH" \
                 --model "$M2" \
                 --gpu_index 1 \
                 > ./result/"$M2"_"$EPOCH".log 2>&1

wait

# gru
python3 train.py --epoch  "$EPOCH" \
                 --model "$M3" \
                 --gpu_index 0 \
                 > ./result/"$M3"_"$EPOCH".log 2>&1 & 

# tcn
python3 train.py --epoch  "$EPOCH" \
                 --model "$M4" \
                 --gpu_index 0 \
                 > ./result/"$M4"_"$EPOCH".log 2>&1
wait

echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"