# data assimilation py
データ同化でよく使うpythonモジュール．自分の開発用．
開発中のコードも多い．

## install
```
pip install git+https://github.com/KotaTakeda/da_py.git
```

### Commonly use
- jupyter-notebook
- matplotlib
- seaborn
- plotly
- pandas
- tqdm
- pot


## OSSE settings
### Example 1
```py
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
*P. J. Leeuwen, Y. Cheng, and S. Reich, Nonlinear Data Assimilation,*
```py
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
```py
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
    - alpha = 1.0 ~ 1.1
    - rho = 5 ~ 10 by 1
``````