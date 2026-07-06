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
from da.exkf import ExKF
from da.pf import ParticleFilter
from da.l63 import lorenz63
from da.l96 import lorenz96
from da.scheme import rk4
```

## Inflation convention

Multiplicative inflation parameters named `alpha` are anomaly inflation factors:

```text
A -> alpha * A
Pf -> alpha^2 * Pf
```

This convention applies to `ETKF`, `LETKF`, and the non-additive `PO` update. In `PO(additive_inflation=True)`, `alpha` remains an additive covariance inflation amplitude.

## Examples

`examples/` は代表例を次の3層に整理しています。

- `examples/scripts/`: CI で実行可能な軽量 benchmark script。
- `examples/notebooks/`: 数式説明つきの最小 tutorial notebook。
- `examples/archive/`: 旧 notebook/script/PDF/配列データの退避先。研究履歴として保持しますが、現 API との同期対象ではありません。

代表例メタデータの運用上の正本は `examples/example_registry.json` です。数式記号は `docs/notation.md`、代表設定一覧は `docs/examples.md` を参照してください。

代表 script:

- `examples/scripts/l63_3dvar_exkf.py`
- `examples/scripts/l63_pf.py`
- `examples/scripts/l63_etkf.py`
- `examples/scripts/l63_etpf.py` (`POT` がない環境では skip)
- `examples/scripts/l96_etkf.py`
- `examples/scripts/l96_letkf.py`
- `examples/scripts/nse2d_etkf.py`

```sh
python examples/scripts/l63_etkf.py
python examples/scripts/l96_letkf.py
python examples/scripts/nse2d_etkf.py
```

将来的には QMC、Henon、L96 PF、NSE2D partial-observation/synchronization、
GitHub Pages へ埋め込む notebook tutorial を整備する予定です。

## Visualization

図の作成は `da.viz` にまとめています。これは `KotaTakeda/research-design-system`
の汎用作図レイヤ `research_figures`(publication 用 `mplstyle`、プロット種別ごとの
プリミティブ、パネルレイアウト、原子的な保存ユーティリティ)を
`da.viz._research_figures` に**ベンダリング(同梱)**した薄いアダプタです。
共有 API を*利用*しつつ、`research-design-system` への実行時依存を持たずに
`examples` extra を self-contained に保つ方針です(同梱物の由来は
`src/da/viz/_research_figures/_VENDORED.md` を参照)。

- 汎用プリミティブ・レイアウト・スタイル・保存関数は `da.viz` から
  そのまま再エクスポートしています(`line_plot`, `image_plot`, `multi_panel`,
  `shared_colorbar`, `panel_labels`, `save_png`, `style_context` など)。
- 渦度など符号付き場のカラーマップは `da.viz.vorticity_cmap` を使います
  (`seaborn` があれば `icefire`、無ければ `coolwarm` に自動フォールバックし、
  `seaborn` は必須ではありません)。
- カラーサイクルのデフォルトは `earth_muted_natural`
  (`da.viz.DEFAULT_COLOR_CYCLE`)です。`style_context` / `single_panel` /
  `multi_panel` などの `cycle=` 引数で他のパレット
  (`viz.list_color_cycles()` 参照)や色リストに変更でき、`cycle=None` で
  publication スタイル本来のサイクルに戻せます。
- 渦度パネルや RMSE 曲線などドメイン固有の作図関数は `da_py` 側
  (`da.visualize` や `examples/`)に置き、共有レイヤには数値・DA ロジックを
  一切移しません。

```py
from da import viz

with viz.style_context():
    ax = viz.line_plot(times, rmse, label="ETKF")
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE")
    ax.legend()
    viz.save_png(ax.figure, "data/rmse.png")
```

`matplotlib` + `numpy` があれば図を生成でき、`seaborn` は任意です。作例は
`examples/scripts/l96_etkf.py`, `examples/scripts/nse2d_etkf.py` を参照してください。

2D NSE の DA 検証は段階的に行います。ETKF の前段として、低モード観測のみで
同期するか(直接挿入/ナッジング)を `examples/archive/nse2d_synchronization.py` で
検証できます(`NSE2DTorus.project_low_modes` / `project_high_modes` を使用)。
その後段の Kelly 型 ETKF ベンチマーク(自由発展/全モード/低モード/
高モードのみ観測の比較)は `examples/archive/nse2d_partial_obs_enkf.py` を参照してください。

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
    - alpha = 1.0 ~ 1.1  # anomaly inflation
  - LETKF
    - m = 8 ~ 20 by 4
    - alpha = 1.0 ~ 1.1 by 0.02  # anomaly inflation
    - rho = 5 ~ 10 by 1
```
