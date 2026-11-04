#!/usr/bin/env bash
# Scorer-agreement data: three passes over the validation papers with 10 generators, scored by
# the ground truth proxy (Sonnet 4.6, paper-aware) and the weak scorer (Haiku 4.5, blind).
#
# The variance decomposition needs three runs that share tasks and generators:
#   A  generate + score with both judges
#   B  rescore run A's SAVED proposals with fresh judge calls   -> isolates judge noise
#   C  generate AGAIN from scratch + score                      -> isolates generator noise
#   C_haiku  rescore run C with the weak scorer only            -> weak-score C column
#
# Cost warning. Each generate pass is 10 generators x ~304 papers, and every sample is
# scored by Sonnet with a full paper (~40k tokens) in context.
#
# Usage:
#   bash run_data.sh A          # or B | C | C_haiku | all
#   N_PAPERS=5 bash run_data.sh all       # cheap smoke test
#
# Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY

set -uo pipefail
cd "$(dirname "$0")/../.."

# Papers to evaluate on. The paper uses the held-out spotlight/oral validation set.
# Override for a smoke test on a small corpus, e.g. PAPER_IDS=paper_ids/train_posters.txt
PAPER_IDS="${PAPER_IDS:-paper_ids/val_spotlight_oral.txt}"
EVAL="eval_harness/experiments_quality_eval.py"
LOGROOT="${LOGROOT:-logs/scorer_agreement}"
MAX_CONN="${MAX_CONN:-64}"
N_PAPERS="${N_PAPERS:-0}"     # 0 = all papers in the id list

DIR_A="$LOGROOT/runA"
DIR_B="$LOGROOT/runB"
DIR_C="$LOGROOT/runC"
DIR_C_HAIKU="$LOGROOT/runC_haiku"

mkdir -p "$DIR_A" "$DIR_B" "$DIR_C" "$DIR_C_HAIKU"

declare -A REASONING=(
    ["anthropic/claude-haiku-4-5-20251001"]="--reasoning-tokens 8192"
    ["anthropic/claude-opus-4-1"]="--reasoning-tokens 8192"
    ["anthropic/claude-opus-4-6"]="--reasoning-effort high"
    ["openrouter/openai/gpt-oss-20b"]="--reasoning-effort high"
    ["openrouter/openai/gpt-oss-120b"]="--reasoning-effort high"
    ["openai/gpt-5.4-nano"]="--reasoning-effort high"
    ["openai/gpt-5.4-mini"]="--reasoning-effort high"
    ["openai/gpt-5.4"]="--reasoning-effort high"
    ["google/gemini-3.1-pro-preview"]="--reasoning-effort high"
    ["google/gemini-3.1-flash-lite-preview"]="--reasoning-effort high"
)

# Cohorts mix providers so we do not hammer one rate limit. 4 in parallel x 64 conns.
COHORT_1=(
    "anthropic/claude-haiku-4-5-20251001"
    "openrouter/openai/gpt-oss-20b"
    "openai/gpt-5.4-nano"
    "google/gemini-3.1-pro-preview"
)
COHORT_2=(
    "anthropic/claude-opus-4-1"
    "openrouter/openai/gpt-oss-120b"
    "openai/gpt-5.4-mini"
    "google/gemini-3.1-flash-lite-preview"
)
COHORT_3=(
    "anthropic/claude-opus-4-6"
    "openai/gpt-5.4"
)

# $1=model $2=logdir $3=include weak scorer (0|1)
run_gen() {
    local model="$1" logdir="$2" inc_weak="$3"
    local roles="--model-role grader=anthropic/claude-sonnet-4-6"
    [[ "$inc_weak" == "1" ]] && roles="$roles --model-role grader2=anthropic/claude-haiku-4-5-20251001"
    echo "[$(date +%H:%M:%S)] >> generate $model -> $logdir"
    uv run inspect eval "$EVAL@experiments_quality" \
        --model "$model" \
        ${REASONING[$model]} \
        -T paper_ids="$PAPER_IDS" \
        -T use_paper=False \
        -T paper_grader_roles=grader \
        -T judge_reasoning_effort=high \
        -T judge_reasoning_tokens=8192 \
        -T post_cutoff_only=True \
        -T use_classifications=True \
        -T limit="$N_PAPERS" \
        -T prompt_style=minimal \
        $roles \
        --max-connections "$MAX_CONN" \
        --log-dir "$logdir" \
        2>&1 | tail -5 \
        || echo "[$(date +%H:%M:%S)] WARN: $model failed; continuing"
}

# $1=source .eval log $2=logdir $3=which judges (both|weak)
run_rescore() {
    local gen_log="$1" logdir="$2" which="$3"
    local roles
    if [[ "$which" == "weak" ]]; then
        roles="--model-role grader=anthropic/claude-haiku-4-5-20251001"
    else
        roles="--model-role grader=anthropic/claude-sonnet-4-6 --model-role grader2=anthropic/claude-haiku-4-5-20251001"
    fi
    echo "[$(date +%H:%M:%S)] >> rescore $(basename "$gen_log") -> $logdir ($which)"
    local extra=""
    [[ "$which" == "both" ]] && extra="-T paper_grader_roles=grader"
    uv run inspect eval "$EVAL@rescore_quality" \
        -T gen_logs="$gen_log" \
        -T use_paper=False \
        $extra \
        -T judge_reasoning_effort=high \
        -T judge_reasoning_tokens=8192 \
        $roles \
        --max-connections "$MAX_CONN" \
        --log-dir "$logdir" \
        2>&1 | tail -5 \
        || echo "[$(date +%H:%M:%S)] WARN: rescore $gen_log failed; continuing"
}

generate_phase() {
    local logdir="$1" inc_weak="$2" label="$3"
    echo "================================================================"
    echo "  $label -> $logdir"
    echo "================================================================"
    for cohort in COHORT_1 COHORT_2 COHORT_3; do
        eval "models=( \"\${${cohort}[@]}\" )"
        echo "[$(date +%H:%M:%S)] -- $cohort (${#models[@]} in parallel) --"
        local pids=()
        for m in "${models[@]}"; do
            run_gen "$m" "$logdir" "$inc_weak" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
    done
}

rescore_phase() {
    local src="$1" dst="$2" which="$3"
    echo "================================================================"
    echo "  rescore $src -> $dst ($which)"
    echo "================================================================"
    local logs=("$src"/*.eval)
    if [[ ${#logs[@]} -eq 0 || ! -f "${logs[0]}" ]]; then
        echo "ERROR: no source logs in $src. Run the generate phase first."
        return 1
    fi
    local i=0
    for log in "${logs[@]}"; do
        run_rescore "$log" "$dst" "$which" &
        i=$((i+1))
        (( i % 4 == 0 )) && wait
    done
    wait
}

case "${1:-all}" in
    A)       generate_phase "$DIR_A" 1 "Run A: generate + both judges" ;;
    C)       generate_phase "$DIR_C" 0 "Run C: regenerate + ground truth proxy only" ;;
    B)       rescore_phase "$DIR_A" "$DIR_B" both ;;
    C_haiku) rescore_phase "$DIR_C" "$DIR_C_HAIKU" weak ;;
    all)
        generate_phase "$DIR_A" 1 "Run A: generate + both judges"
        generate_phase "$DIR_C" 0 "Run C: regenerate + ground truth proxy only"
        rescore_phase "$DIR_A" "$DIR_B" both
        rescore_phase "$DIR_C" "$DIR_C_HAIKU" weak
        ;;
    *) echo "Unknown phase: $1  (use A | B | C | C_haiku | all)"; exit 1 ;;
esac

echo ""
echo "[$(date +%H:%M:%S)] === done. Next: uv run python figures/scorer_agreement/extract.py ==="
