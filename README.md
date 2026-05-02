# Climate Change Impact in Solar PV Output over Indonesia
This project is written to complete my thesis final defense.  I am still working on it, so there must be any updates for codes as well as the analysis. If you guys are interested, any comments or insights will be a great help. 

In this project, I'm working with several climate variables from CMIP6 dataset (such as solar radiation downard/GHI, near-surface temperature, cloud cover, etc.) under two future scenarios to analyze solar PV output over Indonesia. A deep learning model will be used to produce high-resolution daily energy potential of solar PV for historical and future period based on two scenarios (SSP2-4.5 and SSP5-8.5). 

## Requirements
```python
import xarray as xr
import numpy as np
import pandas as pd
import cftime
import os,sys
from scipy import stats
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from mpl_toolkits.basemap import Basemap 
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
```

## Daytime vs Daily Averaged Solar Radiation and Near-surface Temperature
### Global Horizontal Irradiance (GHI)
![Daytime vs Daily GHI](https://github.com/JassLyn1001/solarPV_output_from_CMIP6_datast/blob/main/daytime_vs_daily_solarradiation_allmonth.png)

### Near-surface Temperature (T2M)
![Daytime vs Daily T2M](https://github.com/JassLyn1001/solarPV_output_from_CMIP6_datast/blob/main/daytime_vs_daily_temperature_allmonth.png)

## Spatial Variation 
### Seasonal Climatology of Global Horizontal Irradiance (GHI)
![Seasonal Variation of GHI](https://github.com/JassLyn1001/solarPV_output_from_CMIP6_datast/blob/main/seasonalavg_GHI_daytime.png)

### Seasonal Climatology of Near-surface Temperature (T2M)
![Seasonal Variation of T2M](https://github.com/JassLyn1001/solarPV_output_from_CMIP6_datast/blob/main/seasonalavg_TAS_daytime.png)

### Monthly Climatology of PV Energy Potential 
![Seasonal Variation of PV Cell Energy](https://github.com/JassLyn1001/Longterm_solarPV_output_from_CMIP6Models/blob/main/monclim_PV_energy_per_unit_area_basic.png)
