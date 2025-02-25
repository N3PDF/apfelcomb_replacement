#!/bin/bash

set -euo pipefail

THEORY_ID=40000000
PDF_NAME="NNPDF40_nnlo_as_01180"

LIST_DIS_DATASETS=("HERA_CC_318GEV_EP-SIGMARED")
LIST_HADRONIC_DATASETS=("ATLAS_Z0_7TEV_36PB_ETA")

dis_predictions() {
  THEORYID=$1
  DIS_DATASETS=$2
  NFONLL_ID=$(($THEORYID*100))

  for dataset in "${DIS_DATASETS[@]}"; do
    pineko fonll -c pineko.cli.toml tcards $THEORYID
    pineko fonll -c pineko.cli.toml ekos --overwrite $THEORYID $dataset
    pineko fonll -c pineko.cli.toml fks --overwrite $THEORYID $dataset
    pineko fonll -c pineko.cli.toml combine --overwrite $THEORYID $dataset \
      --FFNS3 $NFONLL_ID \
      --FFN03 $(($NFONLL_ID+1)) \
      --FFNS4zeromass $(($NFONLL_ID+2)) \
      --FFNS4massive $(($NFONLL_ID+3)) \
      --FFN04 $(($NFONLL_ID+4)) \
      --FFNS5zeromass $(($NFONLL_ID+5)) \
      --FFNS5massive $(($NFONLL_ID+6))
  done

  # Compare the DIS FK tables with the Grids
  grids=(./theory_productions/data/grids/"$NFONLL_ID"/*.pineappl.lz4)
  for gridpath in "${grids[@]}"; do
    gridname=$(basename "$gridpath")
    for idx in {0..6}; do
      NFONLL_IDX=$((NFONLL_ID+idx))
      pineko compare ./theory_productions/data/fktables/"$NFONLL_IDX"/"$gridname" \
        ./theory_productions/data/grids/"$NFONLL_IDX"/"$gridname" 2 0 \
        $PDF_NAME --threshold 2000 # TODO: Check what is happening here
    done
  done
}

hadronic_predictions() {
  THEORYID=$1
  HADRONIC_DATASETS=$2

  for dataset in "${HADRONIC_DATASETS[@]}"; do
    pineko theory -c pineko.cli.toml opcards --overwrite $THEORYID $dataset
    pineko theory -c pineko.cli.toml ekos --overwrite $THEORYID $dataset
    pineko theory -c pineko.cli.toml fks --overwrite $THEORYID $dataset
  done

  # Compare the Hadronic FK tables with the Grids
  grids=(theory_productions/data/grids/"$THEORYID"/*.pineappl.lz4)
  for gridpath in "${grids[@]}"; do
    gridname=$(basename "$gridpath")
    pineko compare ./theory_productions/data/fktables/"$THEORYID"/"$gridname" \
      ./theory_productions/data/grids/"$THEORYID"/"$gridname" 2 0 \
      $PDF_NAME --threshold 2
  done
}

compare_predictions() {
  REFERED_FK=$1
  CURRENT_FK=$2

  # Extract the predictions - the last column
  diffs=($(pineappl diff $REFERED_FK $CURRENT_FK $PDF_NAME | awk 'NR>2 {print $NF}'))

  preds_length=${#diffs[@]} # Get the length of the predictions
  for ((bin=0; bin<preds_length; bin++)); do
    pred_value=${diffs[bin]}
    value=$(printf "%.16f" "$pred_value") # Make sure it is in float representation
    # https://www.shell-tips.com/bash/math-arithmetic-calculation/#gsc.tab=0
    abs_diff=$(echo "scale=10; if ($value< 0) -($value) else $value" | bc)
    check_diff=$(echo "$abs_diff > 0.005" | bc) # Set threshold to 5 permille

    if [[ $check_diff -eq 1 ]]; then
      echo "Bin $bin: ($REFERED_FK) and ($CURRENT_FK) differ more than 5 permille."
      exit 1
    fi
  done
}

compare_fks_with_reference() {
  THEORYID=$1
  fktables=(./theory_productions/data/fktables/"$THEORYID"/*.pineappl.lz4)
  for fktable_path in "${fktables[@]}"; do
    fkname=$(basename "$fktable_path")
    fkref="./theory_productions/data/fktables/$THEORYID/$fkname"
    fkcur="./theory_productions/data/fktables/$THEORYID/$fkname" # TODO
    compare_predictions "$fkref" "$fkcur"
  done
}

dis_predictions $THEORY_ID $LIST_DIS_DATASETS
hadronic_predictions $THEORY_ID $LIST_HADRONIC_DATASETS
compare_fks_with_reference $THEORY_ID
