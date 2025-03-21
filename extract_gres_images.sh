#!/bin/bash

echo "Aggregating GRES Images from XVIEW1 Data"
set -e

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
PROJECT_NAME="lisat_data"
SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPTS_DIR}" # "$(realpath "${SCRIPTS_DIR}"/../../../../)"
ENCLAVE_SCRIPTS_ROOT="$(realpath "${SCRIPTS_DIR}"/../${PROJECT_NAME}/src/)"
PATH_TO_XVIEW1_IMAGES="${1:-/datasets/xview1_2024-01-10_1815/train_images}"
PATH_TO_XVIEW1_GEOJSON="${2:-/datasets/xview1_2024-01-10_1815/xView_train.geojson}"

OUTPUT_DIR_BASE="${3:-.}"

SLICE_SIZE="${4:-512}"
OVERLAP="${5:-0}"
PYTHON_ENV_NAME="lisat_data"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/chipped_data/${SLICE_SIZE}_${OVERLAP}"

echo ""
echo "========================================================================"
echo "create_env.sh"
echo "SCRIPTS_DIR: ${SCRIPTS_DIR}"
echo "PROJ_ROOT: ${PROJ_ROOT}"
echo "ENCLAVE_SCRIPTS_ROOT: ${ENCLAVE_SCRIPTS_ROOT}"
echo "PYTHON_ENV_NAME: ${PYTHON_ENV_NAME}"
echo "PATH_TO_XVIEW1_IMAGES: ${PATH_TO_XVIEW1_IMAGES}"
echo "PATH_TO_XVIEW1_GEOJSON: ${PATH_TO_XVIEW1_GEOJSON}"
echo "SLICE_SIZE: ${SLICE_SIZE}"
echo "OVERLAP: ${OVERLAP}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "========================================================================"

# Load conda/mamba
if [ -d ~/anaconda3/etc/profile.d ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -d /opt/miniconda-latest/etc/profile.d ]; then
    source /opt/miniconda-latest/etc/profile.d/conda.sh
elif [ -d ~/miniconda/etc/profile.d ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -d ~/mambaforge/etc/profile.d ]; then
    source ~/mambaforge/etc/profile.d/conda.sh
    source ~/mambaforge/etc/profile.d/mamba.sh
else
    echo "ERROR, no conda installation found"
    exit 1
fi

# Activate conda environment:
conda activate "${PYTHON_ENV_NAME}"

# Script settings
export NUMEXPR_MAX_THREADS=128

# xview to coco
python "${ENCLAVE_SCRIPTS_ROOT}/xview/xview_to_coco.py" \
    --train_images_dir="${PATH_TO_XVIEW1_IMAGES}" \
    --train_geojson_path="${PATH_TO_XVIEW1_GEOJSON}" \
    --output_dir="${OUTPUT_DIR}" \
    --train_split_rate=1.0

# Rename train.json to xview_coco.json
mv "${OUTPUT_DIR}/train.json" "${OUTPUT_DIR}/xview_coco_full.json"
rm "${OUTPUT_DIR}/val.json"

# Do the split using selected seed from split_search.py results:
python "${ENCLAVE_SCRIPTS_ROOT}/data_scripts/trainvalsplit.py" \
    --val_split_size=0.2 \
    --seed=7782551 \
    --input_json="${OUTPUT_DIR}/xview_coco_full.json" \
    --output_dir="${OUTPUT_DIR}"

# Chip
python "${ENCLAVE_SCRIPTS_ROOT}/xview/slice_xview.py" \
    --image_dir="${PATH_TO_XVIEW1_IMAGES}" \
    --dataset_json_path="${OUTPUT_DIR}/xview_coco_train.json" \
    --output_dir="${OUTPUT_DIR}" \
    --slice_size="${SLICE_SIZE}" \
    --overlap_ratio="${OVERLAP}"

python "${ENCLAVE_SCRIPTS_ROOT}/xview/slice_xview.py" \
    --image_dir="${PATH_TO_XVIEW1_IMAGES}" \
    --dataset_json_path="${OUTPUT_DIR}/xview_coco_val.json" \
    --output_dir="${OUTPUT_DIR}" \
    --slice_size="${SLICE_SIZE}" \
    --overlap_ratio="${OVERLAP}"

# 2 Rename and Merge for better class dist, extract GRES Files

SOURCE_VAL="${OUTPUT_DIR}/xview_coco_val_images_${SLICE_SIZE}_${OVERLAP}"
SOURCE_TRAIN="${OUTPUT_DIR}/xview_coco_train_images_${SLICE_SIZE}_${OVERLAP}"

MERGED_DESTINATION="./gres_images"

DEST_TRAIN="${MERGED_DESTINATION}/train"
DEST_VAL="${MERGED_DESTINATION}/val"
DEST_TEST="${MERGED_DESTINATION}/test"

mapfile -t TRAIN_FILES < ./gres_annotations/train/train.txt
mapfile -t VAL_FILES < ./gres_annotations/val/val.txt
mapfile -t TEST_FILES < ./gres_annotations/test/test.txt

for dir in "$DEST_TRAIN" "$DEST_VAL" "$DEST_TEST"; do
  if [ -d "$dir" ]; then
    rm -rf "$dir"/*
    echo "Cleared existing files in destination directory: $dir"
  else
    mkdir -p "$dir"
    echo "Created destination directory: $dir"
  fi
done

counter=0

copy_and_rename() {
  local source_file=$1
  local destination_dir=$2
  local file_name=$(basename "$source_file")

  counter=$((counter + 1))
  local formatted_counter=$(printf "%012d" "$counter")

  if grep -q "lisat_gres_${formatted_counter}" <<< "${TRAIN_FILES[@]}"; then
    destination_dir="$DEST_TRAIN"
  elif grep -q "lisat_gres_${formatted_counter}" <<< "${VAL_FILES[@]}"; then
    destination_dir="$DEST_VAL"
  elif grep -q "lisat_gres_${formatted_counter}" <<< "${TEST_FILES[@]}"; then
    destination_dir="$DEST_TEST"
  else
    # echo "Counter lisat_gres_${formatted_counter} not found in any annotation file, skipping."
    echo "________________________________________________________________________"
    return
  fi

  local new_name="lisat_gres_${formatted_counter}.jpg"
  local destination_file="$destination_dir/$new_name"

  cp "$source_file" "$destination_file"
  echo "Copied $source_file to $destination_file"
}

echo "Copying and renaming validation files..."
for file in "$SOURCE_VAL"/*; do
  if [ -f "$file" ]; then
    copy_and_rename "$file" "$MERGED_DESTINATION"
  else
    echo "No files found in source validation directory: $SOURCE_VAL"
  fi
done

echo "Copying and renaming training files..."
for file in "$SOURCE_TRAIN"/*; do
  if [ -f "$file" ]; then
    copy_and_rename "$file" "$MERGED_DESTINATION"
  else
    echo "No files found in source training directory: $SOURCE_TRAIN"
  fi
done

echo "Files have been successfully copied and renamed."

rm "${OUTPUT_DIR}/xview_coco_full.json"
rm "${OUTPUT_DIR}/xview_coco_train.json"
rm "${OUTPUT_DIR}/xview_coco_train_512_0.json"
rm "${OUTPUT_DIR}/xview_coco_val.json"
rm "${OUTPUT_DIR}/xview_coco_val_512_0.json"
rm -r "${OUTPUT_DIR}/xview_coco_train_images_512_0"
rm -r "${OUTPUT_DIR}/xview_coco_val_images_512_0"
rm -r "${OUTPUT_DIR_BASE}/chipped_data"
