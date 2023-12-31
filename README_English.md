# Run GraphCast (in AutoDL or other new environments) with one click 
-------------------------------------------------------------------

[README_English.md](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/README_English.md)

[中文说明](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/README.md)

[original_README.md](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/original_README.md)

[original_README_中文](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/original_README_中文.md)


-------------------------------------------------------------------
This project is from the work of Google DeepMind DOI: 10.1126/science.adi2336. original copyright information below:

### License and attribution

The Colab notebook and the associated code are licensed under the Apache
License, Version 2.0. You may obtain a copy of the License at:
https://www.apache.org/licenses/LICENSE-2.0.

The model weights are made available for use under the terms of the Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). You may obtain a copy of the License at:
https://creativecommons.org/licenses/by-nc-sa/4.0/.

The weights were trained on ECMWF's ERA5 and HRES data. The colab includes a few
examples of ERA5 and HRES data that can be used as inputs to the models.
ECMWF data product are subject to the following terms:

1. Copyright statement: Copyright "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)".
2. Source www.ecmwf.int
3. Licence Statement: ECMWF data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/
4. Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.

### Disclaimer

This is not an officially supported Google product.

Copyright 2023 DeepMind Technologies Limited.

### Citation

If you use this work, consider citing our [paper](https://arxiv.org/abs/2212.12794):

```latex
@article{lam2022graphcast,
      title={GraphCast: Learning skillful medium-range global weather forecasting},
      author={Remi Lam and Alvaro Sanchez-Gonzalez and Matthew Willson and Peter Wirnsberger and Meire Fortunato and Alexander Pritzel and Suman Ravuri and Timo Ewalds and Ferran Alet and Zach Eaton-Rosen and Weihua Hu and Alexander Merose and Stephan Hoyer and George Holland and Jacklynn Stott and Oriol Vinyals and Shakir Mohamed and Peter Battaglia},
      year={2022},
      eprint={2212.12794},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
-------------------------------------------------------------------

Related Article:
	[从零开始运行 GraphCast （AutoDL 或者其他新的环境）【 jupyter 示例 】](http://t.csdnimg.cn/3YqvT)

This article provides a hands-on tutorial for running the ****GraphCast****.

Simply click the Jupyter button "Run All," and the code will automatically execute the model's environment installation and operation, perform automatic inference, and demonstrate how to train the model (refer to "III. Running GraphCast from a New Environment" - "5. Run GraphCast"). Depending on the machine, executing all the code may take several minutes to a dozen minutes.

The translated and debugged official ****Jupyter**** example is as follows:

****Running GraphCast from Scratch (AutoDL or other new environments) [Jupyter Example]****

All files can be found at https://github.com/sfsun67/GraphCast-from-Ground-Zero.

## I. Introduction to GraphCast

GraphCast is a weather forecasting system based on machine learning and Graph Neural Networks (GNN). The system has been tested by meteorological agencies, including the European Centre for Medium-Range Weather Forecasts (ECMWF). It is an advanced artificial intelligence model capable of mid-term weather forecasting with unprecedented accuracy. GraphCast can forecast weather conditions up to 10 days in advance, more accurate and faster than the industry's gold standard weather simulation system - High-Resolution Forecast (HRES) produced by the ECMWF.

This model ingeniously uses recursively regular icosahedra for six iterations, generating polyhedra that replace the original Earth latitude and longitude network. Under the same resolution conditions, the number of graph model nodes decreases from one million (1,038,240) to 40,000 (40,962). This allows the model to learn complex data with large-scale multi-features under the GNN framework.
![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/b1ea30dc17314198915e98681ba00499.png#pic_centers=20)

Figure 1: Model Structure

In addition to weather forecasting, GraphCast can open up new directions for other important geographical spatiotemporal forecasting problems, including climate and ecology, energy, agriculture, human and biological activities, and other complex dynamic systems. A learning simulator trained on rich real-world data will play a crucial role in advancing the role of machine learning in physical sciences.

## II. Out-of-the-Box GraphCast

1. Register at AdtoDL.

    [https://www.autodl.com/home](https://www.autodl.com/home)

2. Enter the computing power market.
   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/a381b209ee024b07b81fe2b15af79991.png#pic_center)

3. Choose the graphics card you need. Here, the RTX3090 graphics card is tested. Click "1 Card for Rent" to enter the next interface.
    
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/798ce602c253467580097f371862f022.png#pic_center)

4. Select the "Community Image" tab.
    
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/4e68cb8435db4173982e614af0a46ec1.png#pic_center)

5. Enter "GraphCast," find this project, and create an environment.
    
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/065bde980975458db8ad76490b13d853.png#pic_center)
![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/30ad183cb5e6433dbcaf9c7672d0b912.png#pic_center)

    
6. For model operation, refer to "III. Running GraphCast from a New Environment" - "5. Run GraphCast." Click the Jupyter button "Run All," and the code will automatically execute the model's environment installation and operation. Depending on the machine, executing all the code may take several minutes to a dozen minutes.



## III. Run GraphCast in a New Environment

### 1. Configure the Machine

1. Power on your machine. Here, we demonstrate using AutoDL's RTX 2080 Ti 11GB. The machine comes pre-installed with the deep learning framework: JAX / 0.3.10 / 3.8 (ubuntu18.04) / Cuda 11.1.

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/70244410e1354069866bea5d68fafaed.png#pic_center)

2. If you are using your own machine, ensure that JAX / 0.3.10 / 3.8 (ubuntu18.04) / Cuda 11.1 is pre-installed. GraphCast has not been tested on other versions.
3. If you are using AutoDL, open your preferred IDE, such as VsCode or PyCharm, or directly use AutoDL's provided JupyterLab. Here, we use VsCode and connect remotely to the server. Refer to **AutoDL Help Documentation/VSCode Remote Development** [https://www.autodl.com/docs/vscode/](https://www.autodl.com/docs/vscode/).

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/a5b31c952a4c4bbcaa04e6c0d8eb8a8d.png#pic_center)

4. Configure VsCode.
    1. Install Python in VsCode's extensions, and VsCode will automatically install Pylance.

      ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/52f4e860522a4d03945b961c8921a456.png#pic_center)

    2. Install Jupyter in VsCode's extensions, and VsCode will automatically install Jupyter Cell Tags, Jupyter Cell Tags, and Jupyter Cell Tags.

      ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/ed654008386f416a864ffa24e52627ee.png#pic_center)

    3. At this point, the extensions on the server should look like this.

      ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/931ac2318c9a499e8cf263383906d3e1.png#pic_center)

5. Open the server's root directory.

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/11fef058946949a18956c456c2623034.png#pic_center)

### 2. Clone the Code to the Machine

1. Create two folders in the root directory: code and data.

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/d0aeee4d759041e7b41df22ee6174488.png#pic_center)

2. In the terminal, navigate to the directory: `cd /root/code`

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/deb2876df75f40369098253758fc8754.png#pic_center)

3. Clone the code in the terminal: `git clone [https://github.com/sfsun67/GraphCast-from-Ground-Zero](https://github.com/sfsun67/GraphCast-from-Ground-Zero)`

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/2ac02879fa3b40f7ac42c66bde01692c.png#pic_center)

### 3. Download Data

1. The data here is provided by Google Cloud Bucket ([https://console.cloud.google.com/storage/browser/dm_graphcast](https://console.cloud.google.com/storage/browser/dm_graphcast)). Model weights, normalization statistics, and example inputs can be found on the Google Cloud Bucket. To complete the model training, download the ERA5 dataset from ECMWF.
2. You are free to choose the data you want to test. Note that different data needs to match the model parameters. Here, we provide reference data used in this project:

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/a8c1569ee45a430080fb77c28c8eda69.png#pic_center)


### 4. Dependency Installation

1. Click on the Chinese version Jupyter example and install Python3.10 following the instructions.

   ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/1e6dd666a7c644b880b19248fe0a0a46.png#pic_center)

2. The terminal output will be as follows:

```bash
	   root@autodl-container-48ce11bc52-8d41bf84:~/code# conda update -n base -c defaults conda
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.10.3
  latest version: 23.11.0
...

(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code# python -m ipykernel install --user --name=GraphCast-python3.10
Installed kernelspec GraphCast-python3.10 in /root/.local/share/jupyter/kernels/graphcast-python3.10
(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code#
```
     3. Select the new kernel "GraphCast-python3.10" in Jupyter.

    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/2d42833233ff434cb537ff87c0dd8611.png#pic_center)
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/653579fbb2b6482ebd79db10d564497f.png#pic_center)
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/8d82e782f0a3441883770b3fad5f6279.png#pic_center)

### 5. Run GraphCast

1. Click the "Run All" button in Jupyter; the code will automatically execute the model environment setup and run. Depending on the machine, it may take a few minutes to over ten minutes to complete all the code execution.
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/bc8c8798edbf4968a48a487be167ef53.png#pic_center)

2. The model inference results are as follows:
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/1dc12133bb544c4ba3ab605d306a4783.png#pic_center)
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/cfb0ad46024f4862b8c2ba9244e2e0f5.png#pic_center)

3. The model training results are as follows:
    ![Insert Image Description Here](https://img-blog.csdnimg.cn/direct/ff12aebba2ea4a76b5c656a9e9ad8d60.png#pic_center)




