# init
EPOCH=5
SIZE=25
BATCH=16

mode=train
M0=single
M1=concat
M2=lstm
M3=gru
M4=tcn

echo "Start time: $(TZ=UTC-8 date '+%Y-%m-%d %H:%M:%S')"

# concat
python3 "$mode".py --epoch  "$EPOCH" \
                   --model "$M1" \
                   --gpu_index 2 \
                   --window_size "$SIZE" \
                   --batch_size "$BATCH" \
                   > ./result/"$M1"_"$EPOCH"_"$SIZE".log 2>&1 & 

# lstm 
python3 "$mode".py --epoch  "$EPOCH" \
                   --model "$M2" \
                   --gpu_index 3 \
                   --window_size "$SIZE" \
                   --batch_size "$BATCH" \
                   > ./result/"$M2"_"$EPOCH"_"$SIZE".log 2>&1

wait

# gru
python3 "$mode".py --epoch  "$EPOCH" \
                   --model "$M3" \
                   --gpu_index 2 \
                   --window_size "$SIZE" \
                   --batch_size "$BATCH" \
                   > ./result/"$M3"_"$EPOCH"_"$SIZE".log 2>&1 &

# tcn
python3 "$mode".py --epoch  "$EPOCH" \
                 --model "$M4" \
                 --gpu_index 3 \
                 --window_size "$SIZE" \
                 --batch_size "$BATCH" \
                 > ./result/"$M4"_"$EPOCH"_"$SIZE".log 2>&1

wait

# single frame
python3 "$mode".py --epoch  "$EPOCH" \
                 --model "$M0" \
                 --gpu_index 2 \
                 > ./result/"$M0"_"$EPOCH"_1_v1.log 2>&1

                 
echo "End time: $(TZ=UTC-8 date '+%Y-%m-%d %H:%M:%S')"