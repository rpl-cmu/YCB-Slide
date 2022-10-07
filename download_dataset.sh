#!/bin/bash

mkdir dataset
cd dataset
echo "Downloading the YCB-Slide dataset: sim"
gdown https://drive.google.com/drive/folders/1a-8vfMCkW52BpWOPfqk5WM5zsSjBfhN1?usp=sharing --folder
echo "Downloading the YCB-Slide dataset: real"
gdown https://drive.google.com/drive/folders/1VpMgRxrnerU9Dr6-qgLDPUf3lSrCR_yh?usp=sharing --folder
echo "Downloading the DIGIT backgrounds"
gdown https://drive.google.com/drive/folders/1kYjONLAHRA_j4d2X0Enl-zHBIJFmG7Hc?usp=sharing --folder
echo "Downloading YCB ground-truth models"
gdown --fuzzy https://drive.google.com/file/d/1pAQXSQ3K_mLSegFFHuRg2TvyMIf5wX0d/view?usp=sharing
cd sim
for i in */; do unzip "${i%/}.zip"; done
rm *.zip
cd ../real/
for i in */; do unzip "${i%/}.zip"; done
rm *.zip
cd ..
unzip obj_models.zip
rm obj_models.zip
cd ..
echo "Done!"

