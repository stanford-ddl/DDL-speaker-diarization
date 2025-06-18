#!/bin/bash
# run_video_local.sh: 3D-Speaker video diarization on a local file using pre-downloaded ONNX models.
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -e
#. ./path.sh || exit 1

if [ $# -lt 1 ]; then
    echo "Usage: $0 <local_video.mp4> [<reference.rttm>]"
    exit 1
fi

local_video=$1
local_ref_rttm=$2

stage=1
stop_stage=7

examples=examples
exp=exp_video
conf_file=conf/diar_video.yaml
onnx_dir=pretrained_models   # directory where you have already placed version-RFB-320.onnx, asd.onnx, fqa.onnx, face_recog_ir101.onnx
gpus="0 1 2 3"
nj=4

. egs/3dspeaker/speaker-diarization/local/parse_options.sh || exit 1

# Stage 1: Prepare input lists
mkdir -p $examples
echo "$local_video" > $examples/video.list
if [ -n "$local_ref_rttm" ]; then
    echo "$local_ref_rttm" > $examples/refrttm.list
fi
echo "Stage 1: video list → $examples/video.list"
[ -n "$local_ref_rttm" ] && echo "Stage 1: ref RTTM → $examples/refrttm.list"

# Stage 2: Verify ONNX models & extract raw
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Checking ONNX models in $onnx_dir..."
    if [ ! -d "$onnx_dir" ]; then
        echo "ERROR: pretrained_models directory not found. Please place the ONNX files there."
        exit 1
    fi
    for m in version-RFB-320.onnx asd.onnx fqa.onnx face_recog_ir101.onnx; do
        if [ ! -e "$onnx_dir/$m" ]; then
            echo "ERROR: Missing ONNX model $onnx_dir/$m"
            exit 1
        fi
    done

    echo "Stage 2: Extracting raw video & audio..."
    mkdir -p $exp/raw
    while read -r vf; do
        base=$(basename "$vf" .mp4)
        out_mp4=$exp/raw/$base.mp4
        out_wav=$exp/raw/$base.wav

        [ ! -e "$out_mp4" ] && ffmpeg -nostdin -y -i "$vf" -qscale:v 2 -r 25 "$out_mp4" -loglevel panic
        [ ! -e "$out_wav" ] && ffmpeg -nostdin -y -i "$out_mp4" -ac 1 -ar 16000 -vn "$out_wav" -loglevel panic
    done < $examples/video.list
fi

# generate raw data lists
raw_video_list=$exp/raw/video.list
raw_wav_list=$exp/raw/wav.list
awk '{print "'"$exp"'/raw/"substr($0, match($0, /[^\/]+\.mp4$/))}' $examples/video.list > $raw_video_list
awk '{print "'"$exp"'/raw/"substr($0, match($0, /[^\/]+\.mp4$/), RLENGTH-4)".wav"}' $examples/video.list > $raw_wav_list

# Stage 3: Audio embedding extraction
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Extracting audio embeddings..."
    bash egs/3dspeaker/speaker-diarization/run_audio.sh \
        --stage 2 --stop_stage 4 \
        --examples $exp/raw \
        --exp $exp
fi

# Stage 4: Visual embedding extraction
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Extracting visual embeddings..."
    mkdir -p $exp/embs_video
    torchrun --nproc_per_node=$nj egs/3dspeaker/speaker-diarization/local/extract_visual_embeddings.py \
        --conf $conf_file \
        --videos $raw_video_list \
        --vad $exp/json/vad.json \
        --onnx_dir $onnx_dir \
        --embs_out $exp/embs_video \
        --gpu $gpus \
        --use_gpu
fi

# Stage 5: Clustering & RTTM output
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Clustering embeddings..."
    mkdir -p $exp/rttm
    torchrun --nproc_per_node=$nj egs/3dspeaker/speaker-diarization/local/cluster_and_postprocess.py \
        --conf $conf_file \
        --wavs $raw_wav_list \
        --audio_embs_dir $exp/embs \
        --visual_embs_dir $exp/embs_video \
        --rttm_dir $exp/rttm
fi

# Stage 6: Compute DER if reference provided
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ -n "$local_ref_rttm" ]; then
        echo "Stage 6: Computing DER..."
        python local/compute_der.py \
            --exp_dir $exp \
            --ref_rttm $examples/refrttm.list
    else
        echo "Stage 6: No reference RTTM provided; skipping DER calculation."
    fi
fi

# Stage 7: Generate diarization_result.json in results/
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Output diarization_result.json to results/..."
    mkdir -p results
    torchrun --nproc_per_node=$nj egs/3dspeaker/speaker-diarization/local/out_transcription.py \
        --exp_dir $exp --gpu $gpus \
        --output results/diarization_result.json
fi