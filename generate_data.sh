#!/bin/bash
# Generates all data from given .mat file

function generate_data() {
  random_seed=$(printf "%04d" $1)
  data_name=$2
  train_test=$3

  local logPath="${BASE_DATA_DIR}/${train_test}/${data_name}/01pct/${random_seed}rand/data_log.json"
  echo "Log path: ${logPath}"

  if [[ -e "${logPath}" ]]; then
    echo "Output data already exists, skipping data generation"
  fi
  if [[ "$train_test" = learn ]]; then
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 write_psl_data_snowball.py --seed ${random_seed} --data ${data_name}.mat --learn --closefriend
  else
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 write_psl_data_snowball.py --seed ${random_seed} --data ${data_name}.mat
  fi
}

function main() {
  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <data name>"
    exit 1
  fi

  data_name=$1

  trap exit SIGINT

  generate_data "4212" "${data_name}" "learn"

  for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
    generate_data "${rand_sd}" "${data_name}" "eval"
  done

  return 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
