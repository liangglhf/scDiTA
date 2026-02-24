# **scDiTA: Adaptive Representation Learning from Pre-trained Diffusion Transformer for Single-cell Annotation**

## **1\. Overview**

The scDiTA framework aims to transfer the representational power of generative models to discriminative tasks. To address the high dimensionality and sparsity of scRNA-seq data, we first project raw gene expression profiles into a compact continuous latent space using Principal Component Analysis (PCA). Based on these latent variables, we train a Diffusion Transformer (DiT) using Optimal Transport Flow Matching. This process forces the model to fit the original data distribution, learning the probability flow of cells in the latent space and the complex non-linear dependencies between gene programs.

Once trained, we freeze the DiT weights and use the model as a feature extractor. We input clean latent variables, inject a fixed time step parameter t, and set the classifier guidance to null (Null CFG) for inference. Simultaneously, we introduce a DiT-based feature optimization strategy. This identifies and suppresses massive activations induced by AdaLN that concentrate in specific dimensions, stripping away non-semantic signals related to the generation process. The extracted multi-layer features are adaptively fused via a learnable Scalar Mixing strategy. Finally, they pass through a Low-Rank Adaptation (LoRA) module for lightweight fine-tuning and enter a Multi-Layer Perceptron (MLP) classifier for final cell type annotation.

## **2\. Environment Setup**

If you want to reproduce this project, please first configure the required environment. Run the following commands in your terminal:

```bash
conda create -n scDiTA python=3.8.20 -y  
conda activate scDiTA  
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  
pip install -r requirements.txt
```

## **3\. Demo**

We provide a demo using the Pancreas dataset. We have completed all the preliminary preprocessing, clustering, and PCA dimensionality reduction. The training data is located in `dataset/Pancreas` and the test data is in `test/Pancreas`.

All these files can be generated from the preprocessed `.h5ad` files using the code blocks provided in `pre.ipynb`

To run the demo, please execute the following steps in order:

**Features Extracting**: Load our provided Pancreas DiT weights `DiT/Pancreas/Pancreas.pt` to extract features for both the training and test sets.

```bash
python extract.py train  
python extract.py test
```

**Classifier Training**: Use the extracted training features to train the classifier. You can select your own random seed.

```bash
python classifier\_train.py 0
```

**Cell Type Annotation**: Load the classifier weights obtained in the previous step to annotate the test set features and generate the final results.

```bash
python annotation.py 0
```

If you are running this with a GPU, the entire process takes less than 10 minutes.

## **4\. Running with Custom Data**

If you want to run scDiTA on your own scRNA-seq datasets, please follow these steps:

1. **Preprocessing**: Preprocess your data (including log-normalization and the selection of the top 2000 highly variable genes). `dataset/Pancreas/Pancreas.h5ad` can be a standard reference template.  
2. **Clustering & PCA**: Follow the instructions and code blocks in `pre.ipynb` to perform clustering and PCA dimensionality reduction on your specific training and test sets.  
3. **Train DiT**: Run the following command to train the DiT model from scratch using your preprocessed data.

   ```bash
   python DiT_train.py
   ```
   
4. **Downstream Tasks**: Once the DiT training is complete, follow the exact same steps detailed in the **Demo** section (`extract.py` \-\> `classifier_train.py` \-\> `annotation.py`) to extract features, train  classifier, and do the annotations.
