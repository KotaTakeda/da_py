# da_py

研究・数値実験向けのPythonデータ同化パッケージです。Lorenz-63、
Lorenz-96、2次元Navier-Stokes方程式のモデルと、3DVar、ExKF、
Particle Filter、ETKF、LETKF、ETPF、EnKF-Nを含みます。

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
from da.enkfn import EnKFN
from da.etkfn2011 import ETKFN2011
from da.exkf import ExKF
from da.pf import ParticleFilter
from da.l63 import lorenz63
from da.l96 import lorenz96
from da.scheme import rk4
```

## Examples

`examples/` は代表例を次の3層に整理しています。

- `examples/scripts/`: CI で実行可能な軽量 benchmark script。
- `examples/notebooks/`: 数式説明つきの最小 tutorial notebook。
- `examples/archive/`: 旧 notebook/script/PDF/配列データの退避先。研究履歴として保持しますが、現 API との同期対象ではありません。

代表例メタデータの運用上の正本は
[`examples/example_registry.json`](examples/example_registry.json)です。共通記号は
[`docs/reference/notation.md`](docs/reference/notation.md)、設定と実行方法は
[`docs/guides/examples.md`](docs/guides/examples.md)を参照してください。

| Model | Filter | Script | Notebook |
| --- | --- | --- | --- |
| Lorenz-63 | 3DVar / ExKF | [`l63_3dvar_exkf.py`](examples/scripts/l63_3dvar_exkf.py) | [`l63_3dvar_exkf.ipynb`](examples/notebooks/l63_3dvar_exkf.ipynb) |
| Lorenz-63 | Particle Filter | [`l63_pf.py`](examples/scripts/l63_pf.py) | [`l63_pf.ipynb`](examples/notebooks/l63_pf.ipynb) |
| Lorenz-63 | ETKF | [`l63_etkf.py`](examples/scripts/l63_etkf.py) | [`l63_etkf.ipynb`](examples/notebooks/l63_etkf.ipynb) |
| Lorenz-63 | ETPF | [`l63_etpf.py`](examples/scripts/l63_etpf.py) | [`l63_etpf.ipynb`](examples/notebooks/l63_etpf.ipynb) |
| Lorenz-96 | ETKF | [`l96_etkf.py`](examples/scripts/l96_etkf.py) | [`l96_etkf.ipynb`](examples/notebooks/l96_etkf.ipynb) |
| Lorenz-96 | LETKF | [`l96_letkf.py`](examples/scripts/l96_letkf.py) | [`l96_letkf.ipynb`](examples/notebooks/l96_letkf.ipynb) |
| Lorenz-96 | EnKF-N / tuned ETKF | [`l96_enkfn.py`](examples/scripts/l96_enkfn.py) | [`l96_enkfn.ipynb`](examples/notebooks/l96_enkfn.ipynb) |
| NSE2D | ETKF | [`nse2d_etkf.py`](examples/scripts/nse2d_etkf.py) | [`nse2d_etkf.ipynb`](examples/notebooks/nse2d_etkf.ipynb) |

ETPFにはoptional dependencyのPOTが必要です。各scriptの小規模なデフォルト設定は
動作確認とbenchmarkの把握を目的としており、主要な設定はCLI引数で変更できます。

```sh
python examples/scripts/l63_etkf.py
python examples/scripts/l96_letkf.py
python examples/scripts/nse2d_etkf.py
```


**Inflation convention.**

Multiplicative inflation parameters named `alpha` are anomaly inflation factors:

```text
A -> alpha * A
Pf -> alpha^2 * Pf
```

This convention applies to `ETKF`, `LETKF`, and the non-additive `PO` update. In `PO(additive_inflation=True)`, `alpha` remains an additive covariance inflation amplitude.

The adaptive EnKF-N convention, finite-ensemble derivation, and exact mapping
from its scalar dual objective to the ETKF transform are documented in
[`docs/algorithms/enkfn.md`](docs/algorithms/enkfn.md).


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
