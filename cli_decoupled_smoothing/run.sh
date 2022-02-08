#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG

readonly PSL_VERSION='CANARY-2.3.0'
readonly JAR_PATH="./psl-cli-${PSL_VERSION}.jar"
readonly BASE_NAME='gender'

readonly THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DATA_DIR="${THIS_DIR}/data"
readonly DATA_NAME="Amherst41"
readonly BASE_OUT_DIR="${THIS_DIR}/results"

readonly ADDITIONAL_PSL_OPTIONS='--int-ids -D random.seed=12345 -D log4j.threshold=debug -D log4j.threshold=TRACE --postgres btor'
readonly ADDITIONAL_LEARN_OPTIONS='--learn GaussianProcessPrior -D weightlearning.evaluator=RankingEvaluator -D rankingevaluator.representative=AUROC'
readonly ADDITIONAL_EVAL_OPTIONS='--infer --eval CategoricalEvaluator RankingEvaluator'

readonly RUN_ID='decoupled-smoothing'

function main() {
  trap exit SIGINT

  # Make sure we can run PSL.
  checkRequirements
  fetchPsl

  # eval the data
  for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95; do
    generateData "4212" "learn"
    updateData "learn" "4212" "${pct_lbl}"
    runWeightLearn "$@"

    for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
      generateData "${rand_sd}" "eval"
      updateData "eval" "${rand_sd}" "${pct_lbl}"
      runEvaluation "${pct_lbl}" "$@"
    done
  done
}

function generateData() {
  random_seed=$1
  learn_eval=$3

  printf -v seed_nm "%04d" $random_seed
  local logPath="${BASE_DATA_DIR}/${learn_eval}/${BASE_NAME}/01pct/${seed_nm}rand/data_log.json"

  if [[ -e "${logPath}" ]]; then
    echo "Output data already exists, skipping data generation"
  elif [ "$learn_eval" = learn ]; then
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 ../write_psl_data.py --seed ${random_seed} --data ${data_name}.mat --learn
  else
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 ../write_psl_data.py --seed ${random_seed} --data ${data_name}.mat
  fi
}

function updateData() {
  local learn_eval=$1
  local rand_sd=$2
  local pct_lbl=$3

  sed "s/learn_eval/${learn_eval}/g ; s/rand_sd/${rand_sd}rand/g ; s/pct_lbl/${pct_lbl}pct/g ; s/data_nm/${DATA_NAME}/g" \
      base.data > "gender-${learn_eval}.data"
}

function runWeightLearning() {
  echo "Running PSL Weight Learning"

  java -jar "${JAR_PATH}" --model "${BASE_NAME}.psl" --data "${BASE_NAME}-learn.data" ${ADDITIONAL_LEARN_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@"
  if [[ "$?" -ne 0 ]]; then
    echo 'ERROR: Failed to run weight learning'
    exit 60
  fi
}

function runEvaluation() {
  echo "Running PSL Inference"
  local pct_lbl=$1
  shift

  java -jar "${JAR_PATH}" --model "${BASE_NAME}-learned.psl" --data "${BASE_NAME}-eval.data" --output "inferred-predicates${pct_lbl}" ${ADDITIONAL_EVAL_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@"
  if [[ "$?" -ne 0 ]]; then
    echo 'ERROR: Failed to run infernce'
    exit 70
  fi
}

function checkRequirements() {
  local hasWget
  local hasCurl

  type wget >/dev/null 2>/dev/null
  hasWget=$?

  type curl >/dev/null 2>/dev/null
  hasCurl=$?

  if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
    echo 'ERROR: wget or curl required to download dataset'
    exit 10
  fi

  type java >/dev/null 2>/dev/null
  if [[ "$?" -ne 0 ]]; then
    echo 'ERROR: java required to run project'
    exit 13
  fi
}

function get_fetch_command() {
  type curl >/dev/null 2>/dev/null
  if [[ "$?" -eq 0 ]]; then
    echo "curl -o"
    return
  fi

  type wget >/dev/null 2>/dev/null
  if [[ "$?" -eq 0 ]]; then
    echo "wget -O"
    return
  fi

  echo 'ERROR: wget or curl not found'
  exit 20
}

function fetch_file() {
  local url=$1
  local path=$2
  local name=$3

  if [[ -e "${path}" ]]; then
    echo "${name} file found cached, skipping download."
    return
  fi

  echo "Downloading ${name} file located at: '${url}'."
  $(get_fetch_command) "${path}" "${url}"
  if [[ "$?" -ne 0 ]]; then
    echo "ERROR: Failed to download ${name} file"
    exit 30
  fi
}

# Fetch the jar from a remote or local location and put it in this directory.
# Snapshots are fetched from the local maven repo and other builds are fetched remotely.
function fetchPsl() {
  if [[ $PSL_VERSION == *'SNAPSHOT'* ]]; then
    local snapshotJARPath="$HOME/.m2/repository/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
    cp "${snapshotJARPath}" "${JAR_PATH}"
  else
    local remoteJARURL="https://repo1.maven.org/maven2/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
    fetch_file "${remoteJARURL}" "${JAR_PATH}" 'psl-jar'
  fi
}

main "$@"
