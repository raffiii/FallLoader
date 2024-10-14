#!/bin/bash

# Function to download and extract a zip file from a given URL
download_and_extract() {
    # Extract the filename from the URL
    local URL=$1
    local ZIP_FILE=$(basename $URL)

    # Download the zip file
    echo "Downloading $ZIP_FILE from $URL..."
    curl -O $URL

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful. Extracting $ZIP_FILE..."
        # Extract the zip file
        unzip $ZIP_FILE
        # Check if the extraction was successful
        if [ $? -eq 0 ]; then
            echo "Extraction successful."
        else
            echo "Extraction failed."
        fi
    else
        echo "Download failed."
    fi

    # Remove the zip file regardless of success or failure
    rm -f $ZIP_FILE
    echo "$ZIP_FILE removed."
}

#!/bin/bash

download_videos() {
  # Function to download videos with base URL as a parameter
  base_url=$1

  # Loop over fall-01 to fall-30
  for i in $(seq -w 1 30); do
    # Download cam0 and cam1 for each fall
    wget "$base_url/fall-$i-cam0.mp4"
    wget "$base_url/fall-$i-cam1.mp4"
  done
  for i in $(seq -w 1 40); do
    # Download cam0 and cam1 for each fall
    wget "$base_url/adl-$i-cam0.mp4"
  done
}


# Multiple Cameras Fall Dataset
download_and_extract "http://www.iro.umontreal.ca/~labimage/Dataset/dataset.zip"

# EDF (404: http://sites.google.com/site/occlusiondataset)

# OCCU (404: http://sites.google.com/site/occlusiondataset)

# CAUCAFall
download_and_extract "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7w7fccy7ky-4.zip"

# UR Fall
download_videos "http://fenix.ur.edu.pl/~mkepski/ds/data"

# OOPS (403: https://oops.cs.columbia.edu/data/)

# Le2i (404: https://imvia.u-bourgogne.fr/basededonnees/fall-detection-dataset.html)

# UP Fall
# gdown --id 1JBGU5W2uq9rl8h7bJNt2lN4SjfZnFxmQ # only features csv

# E-FPDS_v2 (Train + Validate + Test)
download_and_extract "https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EXQImG_yi5xOifMZYz79_hcBlxATrYEZP5mCu-li4dcWDw?&Download=1"
download_and_extract "https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EULm_4e4bgBKqnsTxDB5Br4BKf9rApBjYi7T0QrWyJrppw??&Download=1"
download_and_extract "https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EXYxgnEftbtCp2iCgAaWDDQBcAuouxLrV_2kxBDalj3m4w?&Download=1"

# Multi Visual Modality Fall Detection Dataset (contact authors)

# IASLAB-RGBD Fallen Person Dataset
download_and_extract "https://robotics.dei.unipd.it/images/dataset/detection_of_fallen_people/static_sequences.7z"
download_and_extract "https://robotics.dei.unipd.it/images/dataset/detection_of_fallen_people/dynamic_sequences.7z.001"
download_and_extract "https://robotics.dei.unipd.it/images/dataset/detection_of_fallen_people/dynamic_sequences.7z.002"
