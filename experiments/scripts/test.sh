#!/bin/bash
# Usage:
# ./experiments/scripts/rfcn_end2end_ohem.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/rfcn_end2end_ohem.sh 0 ResNet50 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

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
    ITERS=60000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2015_test"
    PT_DIR="coco"
    ITERS=170000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
EXTRA_ARGS_SLUG="couplenet_110000"
LOG="experiments/logs/rfcn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#145 py-RFCN-priv_1 
if false;then
time ./tools/train_net_multi_gpu.py --gpu 4,5,6,7 \
  --solver models/${PT_DIR}/${NET}/couplenet/solver_ohem.prototxt \
  --weights data/imagenet_models/${NET}-model.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/rfcn_end2end_ohem_${PT_DIR}.yml \
  ${EXTRA_ARGS}


set +x
NET_FINAL=`tail -n 100 ${LOG} | grep -B 1 "done solving" | grep "Wrote snapshot" | awk '{print $4}'`
set -x
fi
#NET_FINAL='/data1/gongtao/output/rfcn_mypyramidpsroiforspp40channel_7531concat_1024fc_1024fc_21fc/voc_2007_trainval+voc_2012_trainval/resnet101_rfcn_ohem_iter_60000.caffemodel'
#NET_FINAL='/data1/gongtao/output1/rfcn_mt_gate_roialign_uselr=0_mstrain_coco_2timeitarations/globalnet_iter_1160000_cocoforvoc_rpn5scale.caffemodel'
#NET_FINAL='/data1/gongtao/output/globalnet2_mstrain/voc_2007_trainval+voc_2012_trainval/globalnet_iter_97500.caffemodel'
#NET_FINAL='/data1/gongtao/output/pssnet_cocoforvoc/voc_2007_trainval+voc_2012_trainval/resnet101_rfcn_ohem_iter_30000.caffemodel'
NET_FINAL='/mnt/lvdisk1/gongtao/output/monet_uselr0_release/voc_2007_trainval+voc_2012_trainval/monet_iter_100000.caffemodel'
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/monet/test_agnostic.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/monet_end2end_ohem_${PT_DIR}.yml \
  ${EXTRA_ARGS}

echo "$EXTRA_ARGS_SLUG_60000"

#  --weights data/imagenet_models/${NET}-model.caffemodel \