# Deep Learning-based Multiplex IHC Analysis #

This is the code repository associated with the paper:<br/>
**Deep learning-based image analysis methods for brightfield-acquired multiplex immunohistochemistry images** <br/>
[**Danielle J. Fassler, Shahira Abousamra, Rajarsi Gupta, Chao Chen, Maozheng Zhao, David Paredes, Syeda Areeha Batool, Beatrice S. Knudsen, Luisa Escobar-Hoyos, Kenneth R. Shroyer, Dimitris Samaras, Tahsin Kurc, and Joel Saltz, Deep Learning-based Image Analysis Methods for Brightfield-acquired Multiplex Immunohistochemistry Images, Diagnostic Pathology 15, 100 (2020).**](https://diagnosticpathology.biomedcentral.com/articles/10.1186/s13000-020-01003-0)

Multiplex immunohistochemistry (mIHC) permits the labeling of six or more distinct cell types within a single histologic tissue section. The classification of each cell type requires detection of uniquely colored chromogens localized to cells expressing biomarkers of interest. The most comprehensive and reproducible method to evaluate such slides is to employ digital pathology and image analysis pipelines to whole-slide images (WSIs). Our suite of deep learning tools quantitatively evaluates the expression of six biomarkers in mIHC WSIs. These methods address the current lack of readily available methods to evaluate more than four biomarkers and circumvent the need for specialized instrumentation to spectrally separate different colors. The use case application for our methods is a study that investigates tumor immune interactions in pancreatic ductal adenocarcinoma (PDAC) with a customized mIHC panel.


Notes on repository organization and running:<br/>

1\. NNFramework\_PyTorch and NNFramework\_Pytorch\_external\_call<br/>
Contain model training and testing codes

2\. Other useful scripts are in folders starting with src*

3\. datasets<br/>
Directory for holding downloaded datasets.

4\. trained\_models<br/>
Directory for holding downloaded trained models.

3\. Downloading datasets and trained models:<br/> 
See download_resources.txt<br/>
Contain instructions for downloading the train and test datasets and the trained models.

5\. Running model training and testing:<br/>  
See colorAE_train_n_test.txt and colorUNet_train_n_test.txt<br/>
Contain instructions for training the models.

6\. Combining colorAE and colorUNet predictions: <br/>
See combine_predictions.txt<br/>
Contain instructions for combining the models predictions.

7\. Model evaluation: <br/>
See evaluate_predictions_fscore.txt<br/>
Contain instructions for calculating f-score evaluation as outlined in the paper.

8\. Environment setup:<br/>
Code was tested with python 3.6 and pytorch 1.0.0

### Citation ###
    @article{fassler2020mihc,
    author	=  {Danielle J. Fassler and Shahira Abousamra and Rajarsi Gupta and Chao Chen and Maozheng Zhao and David Paredes and Syeda Areeha Batool, Beatrice S. Knudsen and Luisa Escobar-Hoyos and Kenneth R. Shroyer and Dimitris Samaras and Tahsin Kurc and Joel Saltz},
    title	=  {Deep learning-based image analysis methods for brightfield-acquired multiplex immunohistochemistry images},
    journal	=  {Diagnostic Pathology},
    year	=  {2020},
 	volume	=  {15}
	}



 



