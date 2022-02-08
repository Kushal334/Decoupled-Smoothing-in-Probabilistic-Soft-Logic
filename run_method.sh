#!/bin/bash
# Run a decoupled smoothing method on a single data variation

readonly THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DATA_DIR="${THIS_DIR}/data"
readonly BASE_OUT_DIR="${THIS_DIR}/results"

readonly ADDITIONAL_PSL_OPTIONS='--int-ids -D random.seed=12345 -D log4j.threshold=debug -D log4j.threshold=TRACE --postgres btor'
readonly ADDITIONAL_LEARN_OPTIONS='--learn GaussianProcessPrior -D weightlearning.evaluator=RankingEvaluator -D rankingevaluator.representative=AUROC'
readonly ADDITIONAL_EVAL_OPTIONS='--infer --eval CategoricalEvaluator RankingEvaluator'

# An identifier to differentiate the output of this script/experiment from other scripts.
readonly RUN_ID='decoupled-smoothing'

function display_help() {
  echo "USAGE: $0 <data> <random seed> <percent labeled> {learn|eval} <method dir> ..."
  exit 1
}

function run_psl() {
  local cliDir=$1
  local outDir=$2
  local extraOptions=$3
  local data_nm=$4
  local rand_sd=$5
  local pct_lbl=$6
  local learn_eval=$7

  mkdir -p "${outDir}"

  local outPath="${outDir}/out${pct_lbl}.txt"
  local errPath="${outDir}/out${pct_lbl}.err"
  local timePath="${outDir}/time${pct_lbl}.txt"

  if [[ -e "${outPath}" ]]; then
    echo "Output file already exists, skipping: ${outPath}"
    return 0
  fi

  if [ $learn_eval == 'learn' ]; then
    pushd . >/dev/null
    cd "${cliDir}"

    # fix the data settings
    sed "s/learn_eval/learn/g ; s/rand_sd/${rand_sd}rand/g ; s/pct_lbl/${pct_lbl}pct/g ; s/data_nm/${data_nm}/g" \
      base.data >gender-learn.data

    # Run PSL.
    /usr/bin/time -v --output="${timePath}" ./run-learn.sh ${extraOptions} >"${outPath}" 2>"${errPath}"

    # Copy any artifacts into the output directory.
    # cp -R inferred-predicates "${outDir}/inferred-predicates${pct_lbl}"
    cp *.data "${outDir}/"
    cp *.psl "${outDir}/"
    popd >/dev/null
  else
    pushd . >/dev/null
    cd "${cliDir}"

    # fix the data settings
    sed "s/learn_eval/eval/g ; s/rand_sd/${rand_sd}rand/g ; s/pct_lbl/${pct_lbl}pct/g ; s/data_nm/${data_nm}/g" \
      base.data >gender-eval.data

    # Run PSL.
    /usr/bin/time -v --output="${timePath}" ./run-eval.sh ${extraOptions} >"${outPath}" 2>"${errPath}"

    # Copy any artifacts into the output directory.
    cp -R inferred-predicates "${outDir}/inferred-predicates${pct_lbl}"
    cp *.data "${outDir}/"
    cp *.psl "${outDir}/"
    popd >/dev/null
  fi

}

function run() {
  local exampleDir=$1
  local data_nm=$2
  local rand_sd=$3
  local pct_lbl=$4
  local learn_eval=$5

  local exampleName=$(basename "${exampleDir}")
  local cliDir="$exampleDir"
  local outDir="${BASE_OUT_DIR}/${RUN_ID}/${learn_eval}/${exampleName}/${data_nm}/${rand_sd}"

  local options=""
  case $learn_eval in
  learn)
    options="${ADDITIONAL_LEARN_OPTIONS} ${ADDITIONAL_PSL_OPTIONS}"
    ;;
  eval)
    options="${ADDITIONAL_EVAL_OPTIONS} ${ADDITIONAL_PSL_OPTIONS}"
    ;;
  *)
    options="${ADDITIONAL_PSL_OPTIONS}"
    ;;
  esac

  echo "Running ${exampleName} -- ${RUN_ID}."

  run_psl "${cliDir}" "${outDir}" "${options}" "${data_nm}" "${rand_sd}" "${pct_lbl}" "${learn_eval}"
}

function main() {

  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <data> <random seed> <percent labeled> {learn|eval} <method dir> ..."
    exit 1
  fi

  if [ "$1" == "-h" ]; then
    display_help
  fi

  local data_nm=$1
  shift

  local rand_sd=$(printf "%04d" $1)
  shift

  # TODO make this work with floating point numbers
  local pct_lbl=$1
  shift

  local learn_eval=$1
  shift

  trap exit SIGINT

  echo "data used: ${data_nm} | random seed: ${rand_sd} | percent labeled:${pct_lbl} | train test: ${learn_eval}"

  for exampleDir in "$@"; do
    echo "running: ${exampleDir}"
    run "${exampleDir}" "${data_nm}" "${rand_sd}" "${pct_lbl}" "${learn_eval}" "${i}"
  done
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
