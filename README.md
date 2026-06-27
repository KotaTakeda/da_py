# data assimilation py

データ同化用の Python コードです。Lorenz63、Lorenz96などの低次元力学系と、Particle Filter、ETKF、LETKF、3DVar などのデータ同化手法を含みます。

> 開発中のコードも多いため、研究・実験用途を主な対象としています。

```text
Author: Kota Takeda
License: MIT
```

## Install

GitHub から直接インストールできます。

```sh
pip install git+https://github.com/KotaTakeda/da_py.git
```

examples 用の依存も含める場合は `examples` extra を指定してください。

```sh
pip install "da_py[examples] @ git+https://github.com/KotaTakeda/da_py.git"
```

`etpf` を使う場合は `etpf` extra も指定します。

```sh
pip install "da_py[examples,etpf] @ git+https://github.com/KotaTakeda/da_py.git"
```

ローカルで開発する場合は、リポジトリ直下で editable install してください。開発用依存は `dev` extra に含まれます。

```sh
pip install -e ".[dev]"
```

インストール後の import 名は `da` です。

```py
from da.etkf import ETKF
from da.pf import ParticleFilter
from da.l63 import lorenz63
from da.l96 import lorenz96
from da.scheme import rk4
```

## Examples

`examples/` 以下に Jupyter Notebook と一部 Python script があります。

> Note: examples は開発・実験時の参考コードを含みます。最新バージョンの API と完全に同期しているとは限らないため、実行前に import や引数を確認してください。

主な参考例:

- `examples/l63_etkf.ipynb`
- `examples/l63_pf.ipynb`
- `examples/l96_etkf.ipynb`
- `examples/l96_letkf.ipynb`
- `examples/l96_pf.ipynb`
- `examples/qmc.ipynb`

Python script の参考例:

```sh
python examples/l96_etkf.py
```

## OSSE settings

### Example 1

```text
- Lorenz63
  - s = 10
  - b = 8/3
  - r = 28
- Simulate
  - scheme = RK4
  - dt = 0.01
  - N = 20000
- Obs.
  - r = 1.0
  - H = I_3
  - obs_per = 5
  - spin_up = N/2
- Assimilate
  - PF
    - m = 10 ~ 80 by 10
    - h = 0.0 ~ 0.4 by 0.04
```

### Example 2

Reference: P. J. Leeuwen, Y. Cheng, and S. Reich, *Nonlinear Data Assimilation*.

```text
- Lorenz63
  - s = 10
  - b = 8/3
  - r = 28
- Simulate
  - scheme = implicit midpoint
  - dt = 0.01
  - N = 20000*12 + 200
- Obs.
  - r = 8.0
  - H = [1, 0, 0]
  - obs_per = 12
  - spin_up = 200
- Assimilate
  - PF
    - m = 10 ~ 80 by 10
    - h = 0.0 ~ 0.4 by 0.04
```

### Example 3

```text
- Lorenz96
  - J = 40
  - F = 8
- Simulate
  - scheme = RK4
  - dt = 0.01
  - N = 360*20*2  # 2年分に相当
- Obs.
  - r = 1.0
  - H = I_40
  - obs_per = 5
  - burn_in = N/2
- Assimilate
  - ETKF
    - m = 10 ~ 80 by 10
    - alpha = 1.0 ~ 1.1
  - LETKF
    - m = 8 ~ 20 by 4
    - alpha = 1.0 ~ 1.1 by 0.02
    - rho = 5 ~ 10 by 1
```
