# Run GraphCast (in AutoDL or other new environments) with one click  一键运行 GraphCast （在 AutoDL 或者其他新的环境）
-------------------------------------------------------------------

[README.md]()

[中文说明]()

[original_README.md](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/original_README.md)

[original_README_中文](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/original_README_中文.md)


-------------------------------------------------------------------
本项目来自于 Google DeepMind 的工作 DOI: 10.1126/science.adi2336。原始版权信息如下：
### 许可和归属

Colab笔记本和相关代码的许可为Apache License, Version 2.0。
您可以在以下网址获取许可的副本：https://www.apache.org/licenses/LICENSE-2.0。

模型权重可根据Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)的条款供使用。
您可以在以下

网址获取许可的副本：https://creativecommons.org/licenses/by-nc-sa/4.0/。

权重是在ECMWF的ERA5和HRES数据上训练的。Colab中包含了一些ERA5和HRES数据的示例，可以用作模型的输入。
ECMWF数据产品受以下条款约束：

1. 版权声明：版权 "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)"。
2. 来源 www.ecmwf.int
3. 许可声明：ECMWF数据以Creative Commons Attribution 4.0 International (CC BY 4.0)的方式发布。 https://creativecommons.org/licenses/by/4.0/
4. 免责声明：ECMWF对数据中的任何错误或遗漏、其可用性或由其使用而引起的任何损失或损害概不负责。

### 免责声明

这不是官方支持的Google产品。

版权所有 2023 DeepMind Technologies Limited。

### 引用

如果您使用此工作，请考虑引用我们的[论文](https://arxiv.org/abs/2212.12794)：

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

相关文章：
	[从零开始运行 GraphCast （AutoDL 或者其他新的环境）【 jupyter 示例 】](http://t.csdnimg.cn/3YqvT)

本文提供动手运行 ****GraphCast**** 的教程。

只需点击 Jupyter 按钮 “Run All”，代码将自动执行模型的环境安装和运行，自动推理并演示如何训练模型（参考“三、从新环境运行 GraphCast”-“5. 运行GraphCast”）。依据机器不同，执行完毕所有代码可能需要几分钟到十几分钟。

翻译并调试好的官方 ****jupyter**** 示例如下：

****从零开始运行 GraphCast （AutoDL 或者其他新的环境）【 jupyter 示例 】****

所有文件都可以在 https://github.com/sfsun67/GraphCast-from-Ground-Zero 找到。

## 一、GraphCast 介绍

GraphCast 是一种基于机器学习和图神经网络 (GNN) 的天气预报系统。该系统已被包括欧洲中期天气预报中心（ECMWF） 在内的气象机构测试。这是一种先进的人工智能模型，能够以前所未有的准确度进行中期天气预报。GraphCast 最多可以提前 10 天预测天气状况，比行业黄金标准天气模拟系统 - 由欧洲中期天气预报中心 (ECMWF) 制作的高分辨率预报 (HRES) 更准确、更快。

这种模型巧妙的使用递归的正则二十面体进行六次迭代，所产生的多面体替代原有的地球经纬度网络。在相同分辨率条件下，图模型节点数量从一百万（1, 038, 240）下降到 4 万（40, 962）。使得模型可以在 GNN 框架下学习大规模多特征的复杂数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b1ea30dc17314198915e98681ba00499.png#pic_centers=20)

图1：模型结构

除了天气预报之外，GraphCast 还可以为其他重要的地理时空预报问题开辟新的方向，包括气候和生态、能源、农业、人类和生物活动以及其他复杂的动力系统。在丰富的真实世界数据上训练的学习型模拟器，将在促进机器学习在物理科学中的作用方面发挥关键作用。

## 二、开箱即用的 GraphCast

1. 注册AdtoDL。
    
    [https://www.autodl.com/home](https://www.autodl.com/home)
    
2. 进入算力市场。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a381b209ee024b07b81fe2b15af79991.png#pic_center)



3. 选择你需要的显卡。这里测试 RTX3090 显卡。点击“1卡可租”进入下一界面。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/798ce602c253467580097f371862f022.png#pic_center)

4. 选择“社区镜像”选项卡。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4e68cb8435db4173982e614af0a46ec1.png#pic_center)

5. 输入“GraphCast”，找到本项目，并创建环境。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/065bde980975458db8ad76490b13d853.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/30ad183cb5e6433dbcaf9c7672d0b912.png#pic_center)

    
6. 模型运行参考“三、从新环境运行 GraphCast”-“5. 运行GraphCast”。点击 Jupyter 按钮 “Run All”，代码将自动执行模型的环境安装和运行。依据机器不同，执行完毕所有代码可能需要几分钟到十几分钟。

## 三、从新环境运行 GraphCast

### 1. 配置机器

1. 打开你的机器。这里使用 AutoDL 的RTX 2080 Ti 11GB 进行示范。机器预装深度学习框架： JAX / 0.3.10 / 3.8 (ubuntu18.04) / Cuda 11.1。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/70244410e1354069866bea5d68fafaed.png#pic_center)

2. 如果你使用自己的机器。那么确认预装 JAX / 0.3.10 / 3.8 (ubuntu18.04) / Cuda 11.1 。GraphCast未在其他版本上测试。
3. 如果你使用AutoDL。打开你熟悉的IDE，推荐 VsCode 或者 PyCharm，也直接使用 AutoDL 提供的JupyterLab。这里使用 vscode ，远程连接服务器。详见**AutoDL帮助文档/VSCode远程开发**[https://www.autodl.com/docs/vscode/](https://www.autodl.com/docs/vscode/)。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a5b31c952a4c4bbcaa04e6c0d8eb8a8d.png#pic_center)

4. 配置 VsCode。
    1. 在 VsCode 的拓展中安装 python，同时 VsCode 会自动安装 Pylance。
        
 		![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/52f4e860522a4d03945b961c8921a456.png#pic_center)
    2. 在 VsCode 的拓展中安装 jupyter，同时 VsCode 会自动安装 Jupyter Cell Tags、Jupyter Cell Tags 和 Jupyter Cell Tags。
        ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ed654008386f416a864ffa24e52627ee.png#pic_center)

    3. 此时，服务器中的拓展如下。
        ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/931ac2318c9a499e8cf263383906d3e1.png#pic_center)

5. 打开服务器根目录。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/11fef058946949a18956c456c2623034.png#pic_center)

### 2. 拉取代码到机器

1. 在根目录下创建两个文件夹，分别为 code 和 data 。
    
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d0aeee4d759041e7b41df22ee6174488.png#pic_center)

    
2. 在终端中进入目录 ：cd /root/code
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/deb2876df75f40369098253758fc8754.png#pic_center)

    
3. 在终端中克隆代码 ：git clone [https://github.com/sfsun67/GraphCast-from-Ground-Zero](https://github.com/sfsun67/GraphCast-from-Ground-Zero)
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2ac02879fa3b40f7ac42c66bde01692c.png#pic_center)


### 3. 下载数据

1. 这里的数据由 Google Cloud Bucket ([https://console.cloud.google.com/storage/browser/dm_graphcast](https://console.cloud.google.com/storage/browser/dm_graphcast) 提供。模型权重、标准化统计和示例输入可在Google Cloud Bucket上找到。完整的模型训练需要下载ERA5数据集，该数据集可从ECMWF获得。
2. 可以自由选择想要测试的数据。注意，不同的数据需要和模型参数匹配。这里提供本项目测试所用数据做参考：
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/064b44dc3d474b6aafba43171c5eb3b1.png#pic_center)

    

### 4. 依赖安装

1. 点击中文版 Jupyter 示例，按照说明安装 Python3.10 。 
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1e6dd666a7c644b880b19248fe0a0a46.png#pic_center)

    
2. 终端输出如下：

```bash
	   root@autodl-container-48ce11bc52-8d41bf84:~/code# conda update -n base -c defaults conda
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.10.3
  latest version: 23.11.0

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /root/miniconda3

  added / updated specs:
    - conda


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _openmp_mutex-5.1          |            1_gnu          21 KB  defaults
    brotli-python-1.0.9        |   py38h6a678d5_7         329 KB  defaults
    ca-certificates-2023.12.12 |       h06a4308_0         126 KB  defaults
    certifi-2023.11.17         |   py38h06a4308_0         158 KB  defaults
    cffi-1.16.0                |   py38h5eee18b_0         252 KB  defaults
    charset-normalizer-2.0.4   |     pyhd3eb1b0_0          35 KB  defaults
    conda-package-handling-2.2.0|   py38h06a4308_0         267 KB  defaults
    conda-package-streaming-0.9.0|   py38h06a4308_0          27 KB  defaults
    cryptography-41.0.7        |   py38hdda0065_0         2.0 MB  defaults
    idna-3.4                   |   py38h06a4308_0          93 KB  defaults
    ld_impl_linux-64-2.38      |       h1181459_1         654 KB  defaults
    libffi-3.4.4               |       h6a678d5_0         142 KB  defaults
    libgcc-ng-11.2.0           |       h1234567_1         5.3 MB  defaults
    libgomp-11.2.0             |       h1234567_1         474 KB  defaults
    libstdcxx-ng-11.2.0        |       h1234567_1         4.7 MB  defaults
    ncurses-6.4                |       h6a678d5_0         914 KB  defaults
    openssl-3.0.12             |       h7f8727e_0         5.2 MB  defaults
    pip-23.3.1                 |   py38h06a4308_0         2.6 MB  defaults
    pycosat-0.6.6              |   py38h5eee18b_0          93 KB  defaults
    pycparser-2.21             |     pyhd3eb1b0_0          94 KB  defaults
    pyopenssl-23.2.0           |   py38h06a4308_0          96 KB  defaults
    python-3.8.18              |       h955ad1f_0        25.3 MB  defaults
    readline-8.2               |       h5eee18b_0         357 KB  defaults
    requests-2.31.0            |   py38h06a4308_0          96 KB  defaults
    setuptools-68.2.2          |   py38h06a4308_0         948 KB  defaults
    sqlite-3.41.2              |       h5eee18b_0         1.2 MB  defaults
    tk-8.6.12                  |       h1ccaba5_0         3.0 MB  defaults
    urllib3-1.26.18            |   py38h06a4308_0         196 KB  defaults
    wheel-0.41.2               |   py38h06a4308_0         108 KB  defaults
    xz-5.4.5                   |       h5eee18b_0         646 KB  defaults
    zlib-1.2.13                |       h5eee18b_0         103 KB  defaults
    zstandard-0.19.0           |   py38h5eee18b_0         474 KB  defaults
    ------------------------------------------------------------
                                           Total:        55.8 MB

The following NEW packages will be INSTALLED:

  brotli-python      pkgs/main/linux-64::brotli-python-1.0.9-py38h6a678d5_7
  charset-normalizer pkgs/main/noarch::charset-normalizer-2.0.4-pyhd3eb1b0_0
  conda-package-str~ pkgs/main/linux-64::conda-package-streaming-0.9.0-py38h06a4308_0
  zstandard          pkgs/main/linux-64::zstandard-0.19.0-py38h5eee18b_0

The following packages will be REMOVED:

  brotlipy-0.7.0-py38h27cfd23_1003
  chardet-4.0.0-py38h06a4308_1003
  six-1.16.0-pyhd3eb1b0_0
  tqdm-4.61.2-pyhd3eb1b0_1

The following packages will be UPDATED:

  _openmp_mutex                                   4.5-1_gnu --> 5.1-1_gnu
  ca-certificates                       2021.7.5-h06a4308_1 --> 2023.12.12-h06a4308_0
  certifi                          2021.5.30-py38h06a4308_0 --> 2023.11.17-py38h06a4308_0
  cffi                                1.14.6-py38h400218f_0 --> 1.16.0-py38h5eee18b_0
  conda-package-han~                   1.7.3-py38h27cfd23_1 --> 2.2.0-py38h06a4308_0
  cryptography                         3.4.7-py38hd23ed53_0 --> 41.0.7-py38hdda0065_0
  idna               pkgs/main/noarch::idna-2.10-pyhd3eb1b~ --> pkgs/main/linux-64::idna-3.4-py38h06a4308_0
  ld_impl_linux-64                        2.35.1-h7274673_9 --> 2.38-h1181459_1
  libffi                                     3.3-he6710b0_2 --> 3.4.4-h6a678d5_0
  libgcc-ng                               9.3.0-h5101ec6_17 --> 11.2.0-h1234567_1
  libgomp                                 9.3.0-h5101ec6_17 --> 11.2.0-h1234567_1
  libstdcxx-ng                            9.3.0-hd4cf53a_17 --> 11.2.0-h1234567_1
  ncurses                                    6.2-he6710b0_1 --> 6.4-h6a678d5_0
  openssl                                 1.1.1k-h27cfd23_0 --> 3.0.12-h7f8727e_0
  pip                                 21.1.3-py38h06a4308_0 --> 23.3.1-py38h06a4308_0
  pycosat                              0.6.3-py38h7b6447c_1 --> 0.6.6-py38h5eee18b_0
  pycparser                                       2.20-py_2 --> 2.21-pyhd3eb1b0_0
  pyopenssl          pkgs/main/noarch::pyopenssl-20.0.1-py~ --> pkgs/main/linux-64::pyopenssl-23.2.0-py38h06a4308_0
  python                                  3.8.10-h12debd9_8 --> 3.8.18-h955ad1f_0
  readline                                   8.1-h27cfd23_0 --> 8.2-h5eee18b_0
  requests           pkgs/main/noarch::requests-2.25.1-pyh~ --> pkgs/main/linux-64::requests-2.31.0-py38h06a4308_0
  setuptools                          52.0.0-py38h06a4308_0 --> 68.2.2-py38h06a4308_0
  sqlite                                  3.36.0-hc218d9a_0 --> 3.41.2-h5eee18b_0
  tk                                      8.6.10-hbc83047_0 --> 8.6.12-h1ccaba5_0
  urllib3            pkgs/main/noarch::urllib3-1.26.6-pyhd~ --> pkgs/main/linux-64::urllib3-1.26.18-py38h06a4308_0
  wheel              pkgs/main/noarch::wheel-0.36.2-pyhd3e~ --> pkgs/main/linux-64::wheel-0.41.2-py38h06a4308_0
  xz                                       5.2.5-h7b6447c_0 --> 5.4.5-h5eee18b_0
  zlib                                    1.2.11-h7b6447c_3 --> 1.2.13-h5eee18b_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
requests-2.31.0      | 96 KB     | ########################################################################################## | 100% 
sqlite-3.41.2        | 1.2 MB    | ########################################################################################## | 100% 
ncurses-6.4          | 914 KB    | ########################################################################################## | 100% 
pycosat-0.6.6        | 93 KB     | ########################################################################################## | 100% 
openssl-3.0.12       | 5.2 MB    | ########################################################################################## | 100% 
setuptools-68.2.2    | 948 KB    | ########################################################################################## | 100% 
ca-certificates-2023 | 126 KB    | ########################################################################################## | 100% 
conda-package-stream | 27 KB     | ########################################################################################## | 100% 
python-3.8.18        | 25.3 MB   | ########################################################################################## | 100% 
readline-8.2         | 357 KB    | ########################################################################################## | 100% 
pycparser-2.21       | 94 KB     | ########################################################################################## | 100% 
idna-3.4             | 93 KB     | ########################################################################################## | 100% 
cffi-1.16.0          | 252 KB    | ########################################################################################## | 100% 
certifi-2023.11.17   | 158 KB    | ########################################################################################## | 100% 
tk-8.6.12            | 3.0 MB    | ########################################################################################## | 100% 
conda-package-handli | 267 KB    | ########################################################################################## | 100% 
libstdcxx-ng-11.2.0  | 4.7 MB    | ########################################################################################## | 100% 
pip-23.3.1           | 2.6 MB    | ########################################################################################## | 100% 
charset-normalizer-2 | 35 KB     | ########################################################################################## | 100% 
_openmp_mutex-5.1    | 21 KB     | ########################################################################################## | 100% 
pyopenssl-23.2.0     | 96 KB     | ########################################################################################## | 100% 
ld_impl_linux-64-2.3 | 654 KB    | ########################################################################################## | 100% 
brotli-python-1.0.9  | 329 KB    | ########################################################################################## | 100% 
xz-5.4.5             | 646 KB    | ########################################################################################## | 100% 
cryptography-41.0.7  | 2.0 MB    | ########################################################################################## | 100% 
libffi-3.4.4         | 142 KB    | ########################################################################################## | 100% 
wheel-0.41.2         | 108 KB    | ########################################################################################## | 100% 
urllib3-1.26.18      | 196 KB    | ########################################################################################## | 100% 
libgcc-ng-11.2.0     | 5.3 MB    | ########################################################################################## | 100% 
zstandard-0.19.0     | 474 KB    | ########################################################################################## | 100% 
libgomp-11.2.0       | 474 KB    | ########################################################################################## | 100% 
zlib-1.2.13          | 103 KB    | ########################################################################################## | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
root@autodl-container-48ce11bc52-8d41bf84:~/code# conda create -n GraphCast python=3.10    
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.10.3
  latest version: 23.11.0

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /root/miniconda3/envs/GraphCast

  added / updated specs:
    - python=3.10


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _libgcc_mutex-0.1          |             main           3 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    _openmp_mutex-5.1          |            1_gnu          21 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    bzip2-1.0.8                |       h7b6447c_0          78 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ca-certificates-2023.12.12 |       h06a4308_0         126 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ld_impl_linux-64-2.38      |       h1181459_1         654 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    libffi-3.4.4               |       h6a678d5_0         142 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    libgcc-ng-11.2.0           |       h1234567_1         5.3 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    libgomp-11.2.0             |       h1234567_1         474 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    libstdcxx-ng-11.2.0        |       h1234567_1         4.7 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    libuuid-1.41.5             |       h5eee18b_0          27 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ncurses-6.4                |       h6a678d5_0         914 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    openssl-3.0.12             |       h7f8727e_0         5.2 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    pip-23.3.1                 |  py310h06a4308_0         2.7 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    python-3.10.13             |       h955ad1f_0        26.8 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    readline-8.2               |       h5eee18b_0         357 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    setuptools-68.2.2          |  py310h06a4308_0         957 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    sqlite-3.41.2              |       h5eee18b_0         1.2 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    tk-8.6.12                  |       h1ccaba5_0         3.0 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    tzdata-2023c               |       h04d1e81_0         116 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    wheel-0.41.2               |  py310h06a4308_0         109 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    xz-5.4.5                   |       h5eee18b_0         646 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    zlib-1.2.13                |       h5eee18b_0         103 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ------------------------------------------------------------
                                           Total:        53.5 MB

The following NEW packages will be INSTALLED:

  _libgcc_mutex      anaconda/pkgs/main/linux-64::_libgcc_mutex-0.1-main
  _openmp_mutex      anaconda/pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu
  bzip2              anaconda/pkgs/main/linux-64::bzip2-1.0.8-h7b6447c_0
  ca-certificates    anaconda/pkgs/main/linux-64::ca-certificates-2023.12.12-h06a4308_0
  ld_impl_linux-64   anaconda/pkgs/main/linux-64::ld_impl_linux-64-2.38-h1181459_1
  libffi             anaconda/pkgs/main/linux-64::libffi-3.4.4-h6a678d5_0
  libgcc-ng          anaconda/pkgs/main/linux-64::libgcc-ng-11.2.0-h1234567_1
  libgomp            anaconda/pkgs/main/linux-64::libgomp-11.2.0-h1234567_1
  libstdcxx-ng       anaconda/pkgs/main/linux-64::libstdcxx-ng-11.2.0-h1234567_1
  libuuid            anaconda/pkgs/main/linux-64::libuuid-1.41.5-h5eee18b_0
  ncurses            anaconda/pkgs/main/linux-64::ncurses-6.4-h6a678d5_0
  openssl            anaconda/pkgs/main/linux-64::openssl-3.0.12-h7f8727e_0
  pip                anaconda/pkgs/main/linux-64::pip-23.3.1-py310h06a4308_0
  python             anaconda/pkgs/main/linux-64::python-3.10.13-h955ad1f_0
  readline           anaconda/pkgs/main/linux-64::readline-8.2-h5eee18b_0
  setuptools         anaconda/pkgs/main/linux-64::setuptools-68.2.2-py310h06a4308_0
  sqlite             anaconda/pkgs/main/linux-64::sqlite-3.41.2-h5eee18b_0
  tk                 anaconda/pkgs/main/linux-64::tk-8.6.12-h1ccaba5_0
  tzdata             anaconda/pkgs/main/noarch::tzdata-2023c-h04d1e81_0
  wheel              anaconda/pkgs/main/linux-64::wheel-0.41.2-py310h06a4308_0
  xz                 anaconda/pkgs/main/linux-64::xz-5.4.5-h5eee18b_0
  zlib               anaconda/pkgs/main/linux-64::zlib-1.2.13-h5eee18b_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
libgcc-ng-11.2.0     | 5.3 MB    | ########################################################################################## | 100% 
libstdcxx-ng-11.2.0  | 4.7 MB    | ########################################################################################## | 100% 
ca-certificates-2023 | 126 KB    | ########################################################################################## | 100% 
_openmp_mutex-5.1    | 21 KB     | ########################################################################################## | 100% 
readline-8.2         | 357 KB    | ########################################################################################## | 100% 
ncurses-6.4          | 914 KB    | ########################################################################################## | 100% 
libgomp-11.2.0       | 474 KB    | ########################################################################################## | 100% 
libffi-3.4.4         | 142 KB    | ########################################################################################## | 100% 
tzdata-2023c         | 116 KB    | ########################################################################################## | 100% 
_libgcc_mutex-0.1    | 3 KB      | ########################################################################################## | 100% 
zlib-1.2.13          | 103 KB    | ########################################################################################## | 100% 
libuuid-1.41.5       | 27 KB     | ########################################################################################## | 100% 
wheel-0.41.2         | 109 KB    | ########################################################################################## | 100% 
sqlite-3.41.2        | 1.2 MB    | ########################################################################################## | 100% 
pip-23.3.1           | 2.7 MB    | ########################################################################################## | 100% 
bzip2-1.0.8          | 78 KB     | ########################################################################################## | 100% 
ld_impl_linux-64-2.3 | 654 KB    | ########################################################################################## | 100% 
tk-8.6.12            | 3.0 MB    | ########################################################################################## | 100% 
xz-5.4.5             | 646 KB    | ########################################################################################## | 100% 
setuptools-68.2.2    | 957 KB    | ########################################################################################## | 100% 
python-3.10.13       | 26.8 MB   | ########################################################################################## | 100% 
openssl-3.0.12       | 5.2 MB    | ########################################################################################## | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate GraphCast
#
# To deactivate an active environment, use
#
#     $ conda deactivate

root@autodl-container-48ce11bc52-8d41bf84:~/code# conda init bash && source /root/.bashrc
no change     /root/miniconda3/condabin/conda
no change     /root/miniconda3/bin/conda
no change     /root/miniconda3/bin/conda-env
no change     /root/miniconda3/bin/activate
no change     /root/miniconda3/bin/deactivate
no change     /root/miniconda3/etc/profile.d/conda.sh
no change     /root/miniconda3/etc/fish/conf.d/conda.fish
no change     /root/miniconda3/shell/condabin/Conda.psm1
no change     /root/miniconda3/shell/condabin/conda-hook.ps1
no change     /root/miniconda3/lib/python3.8/site-packages/xontrib/conda.xsh
no change     /root/miniconda3/etc/profile.d/conda.csh
modified      /root/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

+--------------------------------------------------AutoDL--------------------------------------------------------+
目录说明:
╔═════════════════╦════════╦════╦═════════════════════════════════════════════════════════════════════════╗
║目录             ║名称    ║速度║说明                                                                     ║
╠═════════════════╬════════╬════╬═════════════════════════════════════════════════════════════════════════╣
║/                ║系 统 盘║一般║实例关机数据不会丢失，可存放代码等。会随保存镜像一起保存。               ║
║/root/autodl-tmp ║数 据 盘║ 快 ║实例关机数据不会丢失，可存放读写IO要求高的数据。但不会随保存镜像一起保存 ║
║/root/autodl-fs  ║文件存储║一般║可以实现多实例间的文件同步共享，不受实例开关机和保存镜像的影响。         ║
╚═════════════════╩════════╩════╩═════════════════════════════════════════════════════════════════════════╝
CPU ：12 核心
内存：40 GB
GPU ：NVIDIA GeForce RTX 2080 Ti, 1
存储：
  系 统 盘/               ：4% 1.2G/30G
  数 据 盘/root/autodl-tmp：1% 108K/50G
  文件存储/root/autodl-fs ：1% 760M/200G
+----------------------------------------------------------------------------------------------------------------+
*注意: 
1.系统盘较小请将大的数据存放于数据盘或网盘中，重置系统时数据盘和网盘中的数据不受影响
2.清理系统盘请参考：https://www.autodl.com/docs/qa/
(base) root@autodl-container-48ce11bc52-8d41bf84:~/code# conda activate GraphCast
(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code# python -- 
Python 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
[1]+  Stopped                 python --
(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code# python --version
Python 3.10.13
(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code# conda install ipykernel
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.10.3
  latest version: 23.11.0

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /root/miniconda3/envs/GraphCast

  added / updated specs:
    - ipykernel


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    asttokens-2.0.5            |     pyhd3eb1b0_0          20 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    backcall-0.2.0             |     pyhd3eb1b0_0          13 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    comm-0.1.2                 |  py310h06a4308_0          13 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    debugpy-1.6.7              |  py310h6a678d5_0         2.0 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    decorator-5.1.1            |     pyhd3eb1b0_0          12 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    exceptiongroup-1.0.4       |  py310h06a4308_0          28 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    executing-0.8.3            |     pyhd3eb1b0_0          18 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ipykernel-6.25.0           |  py310h2f386ee_0         231 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ipython-8.15.0             |  py310h06a4308_0         1.1 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    jedi-0.18.1                |  py310h06a4308_1         988 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    jupyter_client-8.6.0       |  py310h06a4308_0         185 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    jupyter_core-5.5.0         |  py310h06a4308_0          77 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    libsodium-1.0.18           |       h7b6447c_0         244 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    matplotlib-inline-0.1.6    |  py310h06a4308_0          16 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    nest-asyncio-1.5.6         |  py310h06a4308_0          14 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    packaging-23.1             |  py310h06a4308_0          78 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    parso-0.8.3                |     pyhd3eb1b0_0          70 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    pexpect-4.8.0              |     pyhd3eb1b0_3          53 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    pickleshare-0.7.5          |  pyhd3eb1b0_1003          13 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    platformdirs-3.10.0        |  py310h06a4308_0          33 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    prompt-toolkit-3.0.36      |  py310h06a4308_0         592 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    psutil-5.9.0               |  py310h5eee18b_0         368 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ptyprocess-0.7.0           |     pyhd3eb1b0_2          17 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    pure_eval-0.2.2            |     pyhd3eb1b0_0          14 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    pygments-2.15.1            |  py310h06a4308_1         1.8 MB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    python-dateutil-2.8.2      |     pyhd3eb1b0_0         233 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    pyzmq-25.1.0               |  py310h6a678d5_0         462 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    six-1.16.0                 |     pyhd3eb1b0_1          18 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    stack_data-0.2.0           |     pyhd3eb1b0_0          22 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    tornado-6.3.3              |  py310h5eee18b_0         644 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    traitlets-5.7.1            |  py310h06a4308_0         203 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    wcwidth-0.2.5              |     pyhd3eb1b0_0          26 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    zeromq-4.3.4               |       h2531618_0         331 KB  https://mirrors.ustc.edu.cn/anaconda/pkgs/main
    ------------------------------------------------------------
                                           Total:         9.8 MB

The following NEW packages will be INSTALLED:

  asttokens          anaconda/pkgs/main/noarch::asttokens-2.0.5-pyhd3eb1b0_0
  backcall           anaconda/pkgs/main/noarch::backcall-0.2.0-pyhd3eb1b0_0
  comm               anaconda/pkgs/main/linux-64::comm-0.1.2-py310h06a4308_0
  debugpy            anaconda/pkgs/main/linux-64::debugpy-1.6.7-py310h6a678d5_0
  decorator          anaconda/pkgs/main/noarch::decorator-5.1.1-pyhd3eb1b0_0
  exceptiongroup     anaconda/pkgs/main/linux-64::exceptiongroup-1.0.4-py310h06a4308_0
  executing          anaconda/pkgs/main/noarch::executing-0.8.3-pyhd3eb1b0_0
  ipykernel          anaconda/pkgs/main/linux-64::ipykernel-6.25.0-py310h2f386ee_0
  ipython            anaconda/pkgs/main/linux-64::ipython-8.15.0-py310h06a4308_0
  jedi               anaconda/pkgs/main/linux-64::jedi-0.18.1-py310h06a4308_1
  jupyter_client     anaconda/pkgs/main/linux-64::jupyter_client-8.6.0-py310h06a4308_0
  jupyter_core       anaconda/pkgs/main/linux-64::jupyter_core-5.5.0-py310h06a4308_0
  libsodium          anaconda/pkgs/main/linux-64::libsodium-1.0.18-h7b6447c_0
  matplotlib-inline  anaconda/pkgs/main/linux-64::matplotlib-inline-0.1.6-py310h06a4308_0
  nest-asyncio       anaconda/pkgs/main/linux-64::nest-asyncio-1.5.6-py310h06a4308_0
  packaging          anaconda/pkgs/main/linux-64::packaging-23.1-py310h06a4308_0
  parso              anaconda/pkgs/main/noarch::parso-0.8.3-pyhd3eb1b0_0
  pexpect            anaconda/pkgs/main/noarch::pexpect-4.8.0-pyhd3eb1b0_3
  pickleshare        anaconda/pkgs/main/noarch::pickleshare-0.7.5-pyhd3eb1b0_1003
  platformdirs       anaconda/pkgs/main/linux-64::platformdirs-3.10.0-py310h06a4308_0
  prompt-toolkit     anaconda/pkgs/main/linux-64::prompt-toolkit-3.0.36-py310h06a4308_0
  psutil             anaconda/pkgs/main/linux-64::psutil-5.9.0-py310h5eee18b_0
  ptyprocess         anaconda/pkgs/main/noarch::ptyprocess-0.7.0-pyhd3eb1b0_2
  pure_eval          anaconda/pkgs/main/noarch::pure_eval-0.2.2-pyhd3eb1b0_0
  pygments           anaconda/pkgs/main/linux-64::pygments-2.15.1-py310h06a4308_1
  python-dateutil    anaconda/pkgs/main/noarch::python-dateutil-2.8.2-pyhd3eb1b0_0
  pyzmq              anaconda/pkgs/main/linux-64::pyzmq-25.1.0-py310h6a678d5_0
  six                anaconda/pkgs/main/noarch::six-1.16.0-pyhd3eb1b0_1
  stack_data         anaconda/pkgs/main/noarch::stack_data-0.2.0-pyhd3eb1b0_0
  tornado            anaconda/pkgs/main/linux-64::tornado-6.3.3-py310h5eee18b_0
  traitlets          anaconda/pkgs/main/linux-64::traitlets-5.7.1-py310h06a4308_0
  wcwidth            anaconda/pkgs/main/noarch::wcwidth-0.2.5-pyhd3eb1b0_0
  zeromq             anaconda/pkgs/main/linux-64::zeromq-4.3.4-h2531618_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
packaging-23.1       | 78 KB     | ############################################################################################## | 100% 
matplotlib-inline-0. | 16 KB     | ############################################################################################## | 100% 
ipykernel-6.25.0     | 231 KB    | ############################################################################################## | 100% 
jedi-0.18.1          | 988 KB    | ############################################################################################## | 100% 
jupyter_client-8.6.0 | 185 KB    | ############################################################################################## | 100% 
six-1.16.0           | 18 KB     | ############################################################################################## | 100% 
traitlets-5.7.1      | 203 KB    | ############################################################################################## | 100% 
platformdirs-3.10.0  | 33 KB     | ############################################################################################## | 100% 
pygments-2.15.1      | 1.8 MB    | ############################################################################################## | 100% 
pickleshare-0.7.5    | 13 KB     | ############################################################################################## | 100% 
asttokens-2.0.5      | 20 KB     | ############################################################################################## | 100% 
debugpy-1.6.7        | 2.0 MB    | ############################################################################################## | 100% 
prompt-toolkit-3.0.3 | 592 KB    | ############################################################################################## | 100% 
psutil-5.9.0         | 368 KB    | ############################################################################################## | 100% 
zeromq-4.3.4         | 331 KB    | ############################################################################################## | 100% 
jupyter_core-5.5.0   | 77 KB     | ############################################################################################## | 100% 
decorator-5.1.1      | 12 KB     | ############################################################################################## | 100% 
wcwidth-0.2.5        | 26 KB     | ############################################################################################## | 100% 
ipython-8.15.0       | 1.1 MB    | ############################################################################################## | 100% 
pexpect-4.8.0        | 53 KB     | ############################################################################################## | 100% 
backcall-0.2.0       | 13 KB     | ############################################################################################## | 100% 
comm-0.1.2           | 13 KB     | ############################################################################################## | 100% 
stack_data-0.2.0     | 22 KB     | ############################################################################################## | 100% 
pyzmq-25.1.0         | 462 KB    | ############################################################################################## | 100% 
exceptiongroup-1.0.4 | 28 KB     | ############################################################################################## | 100% 
tornado-6.3.3        | 644 KB    | ############################################################################################## | 100% 
libsodium-1.0.18     | 244 KB    | ############################################################################################## | 100% 
nest-asyncio-1.5.6   | 14 KB     | ############################################################################################## | 100% 
python-dateutil-2.8. | 233 KB    | ############################################################################################## | 100% 
pure_eval-0.2.2      | 14 KB     | ############################################################################################## | 100% 
parso-0.8.3          | 70 KB     | ############################################################################################## | 100% 
ptyprocess-0.7.0     | 17 KB     | ############################################################################################## | 100% 
executing-0.8.3      | 18 KB     | ############################################################################################## | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code# python -m ipykernel install --user --name=GraphCast-python3.10
Installed kernelspec GraphCast-python3.10 in /root/.local/share/jupyter/kernels/graphcast-python3.10
(GraphCast) root@autodl-container-48ce11bc52-8d41bf84:~/code#
```
        
3. 在 Jupyter 中选择新的内核 GraphCast-python3.10。
    
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2d42833233ff434cb537ff87c0dd8611.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/653579fbb2b6482ebd79db10d564497f.png#pic_center)

	![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8d82e782f0a3441883770b3fad5f6279.png#pic_center)

### 5. 运行GraphCast

1. 点击 Jupyter 按钮 “Run All”，代码将自动执行模型的环境安装和运行。依据机器不同，执行完毕所有代码可能需要几分钟到十几分钟。
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/bc8c8798edbf4968a48a487be167ef53.png#pic_center)

    
2. 模型推理结果如下
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1dc12133bb544c4ba3ab605d306a4783.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/cfb0ad46024f4862b8c2ba9244e2e0f5.png#pic_center)

3. 模型训练结果如下
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ff12aebba2ea4a76b5c656a9e9ad8d60.png#pic_center)




