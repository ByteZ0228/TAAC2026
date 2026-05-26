#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

if [[ "${USE_TIME_BUCKETS:-1}" == "0" ]]; then
    TIME_BUCKET_FLAG="--no_time_buckets"
else
    TIME_BUCKET_FLAG="--use_time_buckets"
fi

if [[ "${USE_TIME_PERIOD_FEATS:-1}" == "0" ]]; then
    TIME_PERIOD_FLAG="--no_time_period_feats"
else
    TIME_PERIOD_FLAG="--use_time_period_feats"
fi

if [[ "${USE_FOCAL_LOSS:-0}" == "1" ]]; then
    LOSS_FLAGS="--loss_type focal --focal_alpha 0.1 --focal_gamma 2.0"
else
    LOSS_FLAGS="--loss_type bce"
fi

# ---- Active config: RankMixer NS tokenizer (5 user_int + 1 user_dense + 2 item_int = 8 NS) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --d_model 64 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type transformer \
    --hidden_mult 4 \
    --dropout_rate 0.01 \
    --seq_top_k 50 \
    --rank_mixer_mode full \
    ${LOSS_FLAGS} \
    --num_epochs 6 \
    --patience 100 \
    --eval_every_n_steps 0 \
    --reinit_sparse_after_epoch 1 \
    --reinit_cardinality_threshold 0 \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 10000 \
    "${TIME_BUCKET_FLAG}" \
    "${TIME_PERIOD_FLAG}" \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --d_model 64 \
#     --emb_dim 64 \
#     --num_hyformer_blocks 2 \
#     --num_heads 4 \
#     --seq_encoder_type transformer \
#     --hidden_mult 4 \
#     --dropout_rate 0.01 \
#     --seq_top_k 50 \
#     --rank_mixer_mode full \
#     ${LOSS_FLAGS} \
#     --num_epochs 6 \
#     --patience 100 \
#     --eval_every_n_steps 0 \
#     --reinit_sparse_after_epoch 1 \
#     --reinit_cardinality_threshold 0 \
#     --emb_skip_threshold 1000000 \
#     --seq_id_threshold 10000 \
#     "${TIME_BUCKET_FLAG}" \
#     "${TIME_PERIOD_FLAG}" \
#     --num_workers 8 \
#     "$@"
