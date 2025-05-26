# PGAR
This is the implementation of the paper "Progressive Reasoning based Group Activity Recognition".

## 1. Dependencies
- [environment.yaml](https://github.com/li-lindong/PGAR/blob/main/environment.yaml).

## 2. Datasets
- Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip).
- Download the file `tracks_normalized.pkl` from [cvlab-epfl/social-scene-understanding](https://raw.githubusercontent.com/wjchaoGit/Group-Activity-Recognition/master/data/volleyball/tracks_normalized.pkl).

## 3. Train the Base Model
Fine-tune the base model for the dataset. 
    
    ```
    shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage1.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage1.py
    ```
