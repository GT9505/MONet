#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET="ResNet-101"
NET_lc=${NET,,}
DATASET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=110000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train+coco_2014_val"
    TEST_IMDB="coco_2015_test"
    PT_DIR="coco"
    ITERS=290000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
EXTRA_ARGS_SLUG="monet_release"
LOG="experiments/logs/monet_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/train_net_multi_gpu.py --gpu 0\
  --solver models/${PT_DIR}/${NET}/monet/solver_ohem.prototxt \
  --weights data/imagenet_models/${NET}-model.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/monet_end2end_ohem_${PT_DIR}.yml \
  ${EXTRA_ARGS}


set +x
NET_FINAL=`tail -n 100 ${LOG} | grep -B 1 "done solving" | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/monet/test_agnostic.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/monet_end2end_ohem_${PT_DIR}.yml \
  ${EXTRA_ARGS}

echo "$EXTRA_ARGS_SLUG_60000"