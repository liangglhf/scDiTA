# **scDiTA: Adaptive Representation Learning from Pre-trained Diffusion Transformer for Single-cell Annotation**

## **1\. Overview**

Accurate cell type annotation is the cornerstone of single-cell RNA sequencing (scRNA-seq) data analysis. Yet, it faces severe challenges from high data dimensionality, sparsity, and batch effects. Current discriminative models struggle to capture the intrinsic manifold structure of the data. At the same time, generative models like diffusion models can learn data features precisely, but their use in biology is currently limited mostly to data augmentation.

To bridge this gap, we propose scDiTA. This is an adaptive representation learning framework based on a pre-trained Diffusion Transformer (DiT). It essentially turns the DiT from a generator into a highly efficient feature extractor. In practice, scDiTA fits the cellular topology using a DiT architecture combined with OT-FM. It adaptively fuses multi-layer features through a Learnable Scalar Mixing mechanism. We also use LoRA technology for efficient downstream adaptation. Based on the unique traits of gene expression data, we designed a specific strategy using a fixed large time step and null classifier guidance (Null-CFG). This forces the model into a "deep denoising" state, allowing it to extract robust biological features.

Extensive benchmarking shows that scDiTA significantly outperforms SOTA methods like scRGCL in both accuracy and F1 scores across datasets such as Pancreas, Lung, and Colon. It maintains strong robustness even in extreme scenarios with 50% noise interference or just 10% training samples. Moreover, scDiTA displays excellent out-of-distribution (OOD) detection for unknown cell types. It effectively avoids misclassification by lowering its prediction confidence. The framework is also highly scalable, standing as a solid architecture rather than a mere pile of tricks. Ultimately, scDiTA establishes a new paradigm for single-cell representation learning. It proves that a well-trained generative model is inherently the optimal biological feature extractor.

## **2\. Environment Setup**

Please run the following commands in your terminal:

```bash
conda create -n scDiTA python=3.8.20 -y  
conda activate scDiTA  
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  
pip install -r requirements.txt
```

## **3\. Demo (Pancreas Dataset)**

We provide a demo using the Pancreas dataset. We have completed all the preliminary preprocessing, clustering, and PCA dimensionality reduction. The training data is located in dataset/Pancreas and the test data is in test/Pancreas.

*(Note: All these files can be generated from the preprocessed .h5ad files using the code blocks provided in pre.ipynb)*

To run the demo, please execute the following steps in order:

**Step 1: Extract Features** Load our provided Pancreas DiT weights (DiT/Pancreas/Pancreas.pt) to extract features for both the training and test sets.

python extract.py train  
python extract.py test

**Step 2: Train the Classifier** Use the extracted training features to train the classifier. You can select your own random seed (e.g., 0).

python classifier\_train.py 0

**Step 3: Cell Type Annotation** Load the classifier weights obtained in the previous step to annotate the test set features and generate the final results.

python annotation.py 0

**Performance Tip**: If you are running this with a GPU, the entire process takes less than 10 minutes.

## **4\. Running with Custom Data**

If you want to run scDiTA on your own scRNA-seq datasets, please follow these steps:

1. **Preprocessing**: Preprocess your data (e.g., log-normalization) according to the methods described in our paper. You can use dataset/Pancreas/Pancreas.h5ad as a standard reference template.  
2. **Clustering & PCA**: Follow the instructions and code blocks in pre.ipynb to perform clustering and PCA dimensionality reduction on your specific training and test sets.  
3. **Train DiT**: Run the following command to train the DiT model from scratch using your preprocessed data.  
   python DiT\_train.py

4. **Downstream Tasks**: Once the DiT training is complete, follow the exact same steps detailed in the **Demo** section (extract.py \-\> classifier\_train.py \-\> annotation.py) to extract features, train your classifier, and run the annotations.
