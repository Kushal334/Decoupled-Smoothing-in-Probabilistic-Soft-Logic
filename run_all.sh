#!/bin/bash
# Run a decoupled smoothing method against all data variations.

function main() {
  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <data name> <method cli_dir>"
    exit 1
  fi

  data_name=$1
  method=$2

  trap exit SIGINT

  # eval the data
  for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95; do
    ./run_method.sh "${data_name}" "4212" "${pct_lbl}" "learn" "${method}"

    for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
      ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "eval" "${method}"
    done
  done

  return 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
