# Harvesting Temporal Correlation in Large Vision-Language Models: Using Pose Estimation as a Case Study
###### submitted to Proceedings of ACM International Workshop on Large Vision–Language Model Learning and Applications (LAVA'25)
###### cr: tunyuanchang

[https://lava-workshop.github.io/workshop](https://lava-workshop.github.io/workshop)

#### Training

```
mode=train
EPOCH=10
SIZE=30
BATCH=16
MODEL=[single/concat/lstm/gru/tcn]

python3 "$mode".py --epoch  "$EPOCH" \
                   --model "$MODEL" \
                   --gpu_index "$GPU_INDEX" \
                   --window_size "$SIZE" \
                   --batch_size "$BATCH" \
                   > ./result/"$MODEL"_"$EPOCH"_"$SIZE".log 2>&1

```

#### Testing

```
mode=test
EPOCH=10
SIZE=30
MODEL=[single/concat/lstm/gru/tcn]
CHECKPOINT='model.ckpt'

python3 "$mode".py --model "$MODEL" \
                   --gpu_index "$GPU_INDEX" \
                   --epoch  "$EPOCH" \
                   --window_size "$SIZE" \
                   --checkpoint "$CHECKPOINT"
```