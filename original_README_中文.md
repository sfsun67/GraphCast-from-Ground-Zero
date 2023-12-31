
-------------------------------------------------------------------

[README_English.md](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/README_English.md)

[中文说明](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/README.md)

[original_README.md](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/original_README.md)

[original_README_中文](https://github.com/sfsun67/GraphCast-from-Ground-Zero/blob/main/original_README_中文.md)


-------------------------------------------------------------------

# GraphCast：学习技能娴熟的中程全球天气预报

此软件包包含运行和训练 [GraphCast](https://arxiv.org/abs/2212.12794) 的示例代码。
它还提供了三个预训练模型：

1.  `GraphCast`，GraphCast 论文中使用的高分辨率模型（0.25度
分辨率，37个压力层），训练于1979年至2017年的ERA5数据，

2.  `GraphCast_small`，GraphCast的较小低分辨率版本（1度
分辨率，13个压力层，以及较小的网格），训练于1979年至2015年的ERA5数据，
可用于在内存和计算约束较低的情况下运行模型，

3.  `GraphCast_operational`，高分辨率模型（0.25度分辨率，13个
压力层），预先训练于1979年至2017年的ERA5数据，并在2016年至2021年间进行了微调。
该模型可以从HRES数据初始化（不需要降水输入）。

模型权重、归一化统计数据和示例输入可在[Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast)上找到。

完整的模型训练需要下载
[ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
数据集，可从[ECMWF](https://www.ecmwf.int/)获取。

## 文件概述

最佳起点是在[Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb)中打开`graphcast_demo.ipynb`，其中提供了加载数据、生成随机权重或加载预训练快照、生成预测、计算损失和计算梯度的示例。
GraphCast架构的一步实现在`graphcast.py`中提供。

### 库文件简要描述：

*   `autoregressive.py`：用于运行（和训练）一步GraphCast的包装器
    通过在每一步将输出作为输入自回归地馈送，以JAX可微分的方式生成预测序列。
*   `casting.py`：用于使GraphCast在BFloat16精度下运行的包装器。
*   `checkpoint.py`：用于序列化和反序列化树的工具。
*   `data_utils.py`：用于数据预处理的实用工具。
*   `deep_typed_graph_net.py`：通用的深度图神经网络（GNN）
    该网络在`TypedGraph`上操作，其中输入和输出都是节点和边的特征的平面向量。
    `graphcast.py`中使用这三个分别为Grid2Mesh GNN、Multi-mesh GNN和Mesh2Grid
    GNN。
*   `graphcast.py`：一步预测的主要GraphCast模型架构。
*   `grid_mesh_connectivity.py`：在球面上在规则网格和三角形网格之间转换的工具。
*   `icosahedral_mesh.py`：定义了一个二十面体多网格。
*   `losses.py`：损失计算，包括纬度加权。
*   `model_utils.py`：从输入网格数据生成平面节点和边矢量特征的实用工具，
    以及将节点输出矢量操作回多层网格数据的工具。
*   `normalization.py`：用于一步GraphCast的包装器，用于根据历史值对输入进行标准化，
    并根据历史时间差对目标进行标准化。
*   `predictor_base.py`：定义预测器的接口，GraphCast和所有包装器都实现了该接口。
*   `rollout.py`：类似于`autoregressive.py`，但仅在推理时使用
    使用python循环生成较长但不可微的轨迹。
*   `typed_graph.py`：`TypedGraph`的定义。
*   `typed_graph_net.py`：在`TypedGraph`上定义的简单图神经网络的实现
    这可以组合以构建更深层次的模型。
*   `xarray_jax.py`：使JAX与`xarray`一起工作的包装器。
*   `xarray_tree.py`：与`xarray`一起工作的tree.map_structure的实现。


### 依赖关系。

[Chex](https://github.com/deepmind/chex)，
[Dask](https://github.com/dask/dask)，
[Haiku](https://github.com/deepmind/dm-haiku)，
[JAX](https://github.com/google/jax)，
[JAXline](https://github.com/deepmind/jaxline)，
[Jraph](https://github.com/deepmind/jraph)，
[Numpy](https://numpy.org/)，
[Pandas](https://pandas.pydata.org/)，
[Python](https://www.python.org/)，
[SciPy](https://scipy.org/)，
[Tree](https://github.com/deepmind/tree)，
[Trimesh](https://github.com/mikedh/trimesh) 和
[XArray](https://github.com/pydata/xarray)。


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
