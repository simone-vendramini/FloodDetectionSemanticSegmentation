# Semantic Segmentation on the FloodNet Dataset

This repository contains the code for a project in the "Advanced Machine Learning"
course at the University of Milano-Bicocca. The objective is to implement a semantic
segmentation model on the FloodNet dataset [1].

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Running the Code  

### Step 1: Download and Prepare the Dataset  

1. Download the FloodNet dataset from the link provided below.  
2. Extract the contents into the `FloodNet` directory.  
3. Run the following script to format the dataset:  

   ```bash
   python src/compress.py
   ```

### Step 2: Run the Experiments  

Use the provided Jupyter notebooks to explore, preprocess, and train models. The
notebooks are structured as follows:  

- **`Data-Exploration.ipynb`** – Explore the dataset.  
- **`Data-Augmentation.ipynb`** – Perform data augmentation.  
- **`U-Net.ipynb`** – Implement the U-Net model.  
- **`SegNet.ipynb`** – Implement the SegNet model.  
- **`DeepLab.ipynb`** – Implement the DeepLab model.  
- **`Plots.ipynb`** – Evaluate model performance with visualizations.  

Each notebook is designed to keep the code organized and improve readability,
making it easier to follow the different stages of the project.  

## Dataset

The FloodNet dataset comprises high-resolution aerial images capturing flooded areas.
Each image is annotated with one of ten classes:

0. Background
1. Building - Flooded
2. Building - Non-Flooded
3. Road - Flooded
4. Road - Non-Flooded
5. Water
6. Tree
7. Vehicle
8. Pool
9. Grass

The dataset is divided into three subsets:

- Training set: 1,445 images
- Validation set: 450 images
- Test set: 448 images

Each image is accompanied by a corresponding mask, which is a 3-channel image
where each pixel's integer value represents a class label.

The complete dataset can be downloaded from [FloodNet](https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=3&dl=0).

## Results

![u_net_aum](https://github.com/user-attachments/assets/b37a51bc-b38c-497f-a14b-6ee77037f174)

## References

> [1] Rahnemoonfar, M., Chowdhury, T., Sarkar, A., Varshney, D., Yari, M., & Murphy, R. R. (2021). FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding. IEEE Access, 9, 89644–89654. DOI:10.1109/ACCESS.2021.3090981
