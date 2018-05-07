#!/usr/bin/env bash

DATA_DIR="./ucf101/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}
wget http://crcv.ucf.edu/data/UCF101/UCF101.rar

unrar x UCF101.rar