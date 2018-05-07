#!/usr/bin/env bash

DATA_DIR="./hmdb51/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

mkdir rars && mkdir videos
unrar x hmdb51_org.rar rars/
for rar in $(ls rars); do unrar x "rars/${rar}" videos/; done;
