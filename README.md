# PGAR
This is the implementation of the paper "Progressive Reasoning based Group Activity Recognition".

## 1. Dependencies
- [environment.yaml](https://github.com/li-lindong/PGAR/blob/main/environment.yaml).

## 2. Datasets
- Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip).
- Download the file `tracks_normalized.pkl` from [cvlab-epfl/social-scene-understanding](https://raw.githubusercontent.com/wjchaoGit/Group-Activity-Recognition/master/data/volleyball/tracks_normalized.pkl).

## 3. Train the Base Model
Fine-tune the base model for the dataset. 
    
    ```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage1.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage1.py
    ```

## 4. Train the Final Model
    ```
    # Volleyball dataset
    python scripts/train_volleyball_stage2_dynamic.py

    # Collective Activity dataset
    python scripts/train_collective_stage2_dynamic.py
    ```

If you find our work or the codebase inspiring and useful to your research, please consider ⭐starring⭐ the repo and citing:
```bibtex
@article{PGAR,
  title={Progressive Reasoning based Group Activity Recognition},
  author={Li, Lindong and Qing, Linbo and Tang, Wang and Wang, Pingyu and Gou, Haosong and Zhu, Ce},
  booktitle={Major Revision},
  pages={xx--xx},
  year={2025}
}
```
