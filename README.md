# Comparing classic, deep and semi-supervised learning for whole-transcriptome breast cancer subtyping
Gene expression levels, measuring the transcription activity, are widely used to predict abnormal gene activities, in particular for distinguishing between normal and tumor cells. This problem has been addressed by a variety of machine learning methods; more recently, this problem has been approached using deep learning methods, but they typically failed in meeting the same performance as machine learning. In this paper, we show specific deep learning methods that can achieve similar performance as the best machine learning methods. 

## Source Code
The source code used for the experiments can be found under the "src" directory, with the model classes and scripts to train the models. 
Under "notebooks" one can find several data exploration examples and standalone model experiments for quick prototyping

## Data
To main data sources were used to create this work:
* RNA-Seq data from TCGA - https://portal.gdc.cancer.gov
* RNA_Seq data from TCGA (breast cancer subset, labelled by Ciriello et al.) - http://cbio.mskcc.org/cancergenomics/tcga/brca_tcga/
* RNA-Seq data from the ARCHS4 dataset (breast subset): https://amp.pharm.mssm.edu/archs4/download.html
