# Run GraphCast (in AutoDL or other new environments) with one click  

#一键运行 GraphCast （在 AutoDL 或者其他新的环境）
-------------------------------------------------------------------

[README_English.md](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/README_English.md)

[中文说明](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/README.md)

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
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a8c1569ee45a430080fb77c28c8eda69.png#pic_center)

    

### 4. 依赖安装

1. 点击中文版 Jupyter 示例，按照说明安装 Python3.10 。 
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1e6dd666a7c644b880b19248fe0a0a46.png#pic_center)

    
2. 终端输出如下：

```bash
	   root@autodl-container-48ce11bc52-8d41bf84:~/code# conda update -n base -c defaults conda
Collecting package metadata (current_repodata.json): done
Solving environment: done

...
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




