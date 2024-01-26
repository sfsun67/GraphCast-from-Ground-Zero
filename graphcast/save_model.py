# Copyright 2024 S.F Sune Limited.
# https://github.com/sfsun67/GraphCast-from-Ground-Zero
"""
创建一个数据类，用于存储模型参数和配置信息。
Create a data class to store model parameters and configuration information.
"""

import dataclasses
from typing import Any, Optional, Union


@dataclasses.dataclass
class SubConfig:
  a: int
  b: str


@dataclasses.dataclass
class Config:
  bt: bool
  bf: bool
  i: int
  f: float
  o1: Optional[int]
  o2: Optional[int]
  o3: Union[int, None]
  o4: Union[int, None]
  o5: int | None
  o6: int | None
  li: list[int]
  ls: list[str]
  ldc: list[SubConfig]
  tf: tuple[float, ...]
  ts: tuple[str, ...]
  t: tuple[str, int, SubConfig]
  tdc: tuple[SubConfig, ...]
  dsi: dict[str, int]
  dss: dict[str, str]
  dis: dict[int, str]
  dsdis: dict[str, dict[int, str]]
  dc: SubConfig
  dco: Optional[SubConfig]
  ddc: dict[str, SubConfig]

@dataclasses.dataclass
class ModelConfig:
  resolution: float
  mesh_size: int
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float
  mesh2grid_edge_normalization_factor: float

@dataclasses.dataclass
class TaskConfig:
  input_variables: tuple[str, ...]
  target_variables: tuple[str, ...]
  forcing_variables: tuple[str, ...]
  pressure_levels: tuple[int, ...]
  input_duration: str


@dataclasses.dataclass
class Checkpoint:
  params: dict[str, Any]
  model_config: ModelConfig
  task_config: TaskConfig
  description: str
  license: str



          




