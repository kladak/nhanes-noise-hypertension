#!/bin/bash
# Download the 14 public NHANES XPT files used by the analysis:
# DEMO, AUQ, BPX, BPQ, BMX, SMQ, DIQ for 2011-2012 (_G) and 2015-2016 (_I).
# URL pattern: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/<FILE>_<suffix>.XPT
set -u
mkdir -p data
for pair in "2011 G" "2015 I"; do
  set -- $pair
  year=$1; suffix=$2
  for f in DEMO AUQ BPX BPQ BMX SMQ DIQ; do
    url="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/${year}/DataFiles/${f}_${suffix}.XPT"
    out="data/${f}_${suffix}.XPT"
    if [ -s "$out" ]; then echo "exists $out"; continue; fi
    curl -fL --retry 3 -o "$out" "$url" && echo "downloaded $out"
  done
done
echo "DONE"
