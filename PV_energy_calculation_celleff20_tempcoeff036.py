import xarray as xr
import cftime
import numpy as np
import pandas as pd
import sys, os
from glob import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PATHS = {
    'solar': "/raid/nfs_storageIPB/thesis_JL/copy_of_thesisdata/era5-land/solar_radiation_downward/daysum_00_UTC/",
    'temp': "/raid/nfs_storageIPB/thesis_JL/copy_of_thesisdata/era5-land/near_surface_temp/daymean_00am_11am_UTC/"
}

fname = {
    'solar': "dailysum_seltime_00am_UTC_1981-2021.nc",
    'temp': "dailymean_seltime_00am_11am_UTC_degC_setunit_1980-2021.nc"
}

# PANEL SPECIFICATIONS (615W panel with your parameters)
PANEL_SPECS = {
    'temp_coefficient': 0.0036,   # -0.36%/°C
    'efficiency_STC' : 0.20      # 20% efficiency at STC
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def open_file(filapaths):  # read single files
    with xr.open_dataset(filapaths) as ds:
        print(f"Opened: {filapaths}")
        print(f"Variables: {list(ds.data_vars)}")
    return ds

def reindex_time(original, target):
    """Reindex time dimension and fill the missing values."""
    return (
        original
        .reindex({
            "valid_time": target.get_index('valid_time'),
        })
        .fillna(original.values)
    )

# ============================================================================
# PV CALCULATION FUNCTIONS
# ============================================================================

def temp_cell_calc(T, G):
    """
    Calculate cell temperature based on given air temperature and solar radiation
    
    Parameters:
    -----------
    near-surface temperature (T) : array-like
        Observed/referenced daytime temperature values (in Celsius)
    global solar radiation (G) : array-like
        Observed/referenced total global solar radiation values (in MJ m-2 day-1)
    
    Returns:
    --------
    array : cell temperature (in Celsius)
    """
    c1 = -3.75
    c2 = 1.14
    c3 = 0.0175
    
    return c1 + c2 * T + c3 * G

def cell_efficiency_calc(Tcell, T_ref=25.0, n_ref=0.20, beta=0.0036):
    """
    Calculate PV cell's efficiency as a function of cell temperature
    """
    estimated_efficiency = n_ref * (1 - beta * (Tcell - T_ref))
    return estimated_efficiency

def create_dataset(cell_efficiency, PV_energy_per_m2_kWh, time_coords, spatial_coords, base_attrs):
    """
    Create xarray Dataset with all PV output variables
    
    Parameters:
    -----------
    results : dict
        Dictionary of calculation results
    time_coords : xarray.DataArray
        Time coordinate
    spatial_coords : dict
        Latitude and longitude coordinates
    base_attrs : dict
        Base attributes from your original calculation
    
    Returns:
    --------
    xarray.Dataset : Dataset with PV variable
    """
    
    # Extract coordinates
    coords = {
        'valid_time': time_coords,
        'latitude': spatial_coords['latitude'],
        'longitude': spatial_coords['longitude']
    }
    
    # Create DataArrays for each variable
    data_vars = {}
    
    # 1. Core efficiency and temperature variables 
    data_vars['cell_efficiency'] = xr.DataArray(
        cell_efficiency,
        coords=coords,
        dims=['valid_time', 'latitude', 'longitude'],
        attrs={
            'long_name': 'Temperature-corrected PV Cell Efficiency',
            'units': '1',
            'standard_name': 'pv_cell_efficiency',
            'description': 'PV cell efficiency corrected for operating temperature',
            'reference': '22% at 25°C with -0.30%/°C temperature coefficient'
        }
    )
    
    # 2. Basic PV output (based on cell efficiency and temperature coefficient)
    data_vars['PV_basic_per_m2'] = xr.DataArray(
        PV_energy_per_m2_kWh,
        coords=coords,
        dims=['valid_time', 'latitude', 'longitude'],
        attrs={
            'long_name': 'Basic PV Energy Output per Unit Area',
            'units': 'kWh m-2 day-1',
            'standard_name': 'pv_energy_basic',
            'description': 'Daily PV energy output considering only cell efficiency (no system losses)',
            'calculation': 'GHI_MJ_per_m2 × cell_efficiency ÷ 3.6'
        }
    )
    
    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Update attributes
    ds.attrs.update(base_attrs)
    
    # Add panel specifications
    ds.attrs['panel_efficiency_STC'] = PANEL_SPECS['efficiency_STC']
    ds.attrs['panel_temp_coefficient'] = f"{PANEL_SPECS['temp_coefficient']*100:.2f}%/°C"
    
    return ds

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("PV ENERGY CALCULATION - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    # Read in all data
    print("\n1. Loading data...")
    daytime_temp_ds = open_file(PATHS['temp'] + fname['temp'])
    dailysum_ssrd_ds = open_file(PATHS['solar'] + fname['solar'])
    
    # Data Selection
    print("\n2. Selecting data period (1985-2020)...")
    sel_GHI = dailysum_ssrd_ds.sel(valid_time=slice('1985', '2020'))
    sel_T2M = daytime_temp_ds.sel(valid_time=slice('1985', '2020'))
    
    # Reindexing time dimension
    print("3. Reindexing time dimensions...")
    reindexed_sel_GHI = reindex_time(sel_GHI.ssrd, sel_T2M.t2m)
    
    # Convert units
    print("4. Converting units...")
    GHI_J_per_m2 = reindexed_sel_GHI
    GHI_MJ_per_m2 = GHI_J_per_m2 / 1e6
    
    # Calculate cell temperature and efficiency
    print("5. Calculating PV cell temperature and efficiency...")
    cell_temp_ = temp_cell_calc(sel_T2M.t2m, GHI_MJ_per_m2)
    cell_efficiency = cell_efficiency_calc(
        cell_temp_, 
        T_ref=25.0, 
        n_ref=PANEL_SPECS['efficiency_STC'],  # Use panel's STC efficiency
        beta=PANEL_SPECS['temp_coefficient']  # Use panel's temperature coefficient
    )
    
    # Calculate basic PV output
    print("6. Calculating basic PV energy output...")
    PV_energy_per_m2_kWh = (GHI_MJ_per_m2 * cell_efficiency) / 3.6
    
    # Get coordinates for dataset creation
    time_coords = PV_energy_per_m2_kWh.valid_time
    spatial_coords = {
        'latitude': PV_energy_per_m2_kWh.latitude,
        'longitude': PV_energy_per_m2_kWh.longitude
    }
    
    # Define base attributes
    base_period = '19850101-20201231'
    table_id = 'day'
    
    base_attrs = {
        'title': 'Historical PV Energy Potential over Indonesia',
        'author': 'Jassica Listyarini',
        'institution': 'Department of Geophysics and Meteorology, IPB University',
        'table_id': table_id,
        'period': base_period,
        'data_source': 'ERA5-Land hourly dataset',
        'panel_specifications': f"({PANEL_SPECS['temp_coefficient']*(-1)*100:.2f}%/°C,{PANEL_SPECS['efficiency_STC']*100:.1f}%)", 
        'calculation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create comprehensive dataset
    print("8. Creating comprehensive dataset with multiple output units...")
    comprehensive_ds = create_dataset(
        cell_efficiency,
        PV_energy_per_m2_kWh,
        time_coords=time_coords,
        spatial_coords=spatial_coords,
        base_attrs=base_attrs
    )
    
    # Save comprehensive dataset
    print("\n9. Saving comprehensive dataset...")
    outdir_dir = f"/raid/nfs_storageIPB/thesis_JL/copy_of_thesisdata/interim/PVout"
    path = Path(outdir_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Define save function
    def save_comprehensive_dataset(ds, period):
        """Save comprehensive dataset to NetCDF"""
        try:
            filename = f"PV_energy_updated_id_{table_id}_{period}.nc"
            output_path = path / filename
            
            print(f"Attempting to save to: {output_path}")
            
            # Encoding for different variable types
            encoding = {}
            for var in ds.data_vars:
                encoding[var] = {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 1
                }
            
            # Time encoding
            if 'valid_time' in ds.coords:
                encoding['valid_time'] = {
                    'dtype': 'float64',
                    'units': 'days since 1850-01-01 00:00:00',
                    'calendar': 'standard'
                }
            # Save dataset
            ds.to_netcdf(
                output_path,
                format="NETCDF4",
                engine="netcdf4",
                encoding=encoding,
                unlimited_dims=['valid_time'] if 'valid_time' in ds.dims else None
            )
            print(f"Successfully saved: {filename}")
            return output_path
            
        except Exception as e:
            print(f"Failed to save {filename}: {str(e)}")
            raise
    
    # Save the dataset
    output_path = save_comprehensive_dataset(comprehensive_ds, base_period)
    
    print(f"\nOUTPUT FILE: {output_path}")
    print(f"FILE SIZE: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 70)
    print("CALCULATION COMPLETE")
    print("=" * 70)
    
    #return comprehensive_ds

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Just run it, don't need the object
    main()  # Result discarded
