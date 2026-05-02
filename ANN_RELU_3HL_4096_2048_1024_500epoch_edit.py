import xarray as xr
import cftime
import numpy as np 
import pandas as pd
import seaborn as sns
import re
from glob import glob
from pathlib import Path
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import cmip6_dataprep as prep
import cmip6_dataread as read
# THIS MUST BE THE VERY FIRST CELL IN YOUR NOTEBOOK
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only second GPU
# Or if you want to use all GPUs, specify them explicitly:
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
# Test CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs visible: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# Downscaling PV Energy Output using ANN model
PATHS = {
    'OBS': "/raid/nfs_storageIPB/thesis_JL/copy_of_thesisdata/interim/PVout/",
    'CMIP6': "/raid/nfs_storageIPB/thesis_JL/copy_of_thesisdata/CMIP6/",
    'DEM': "/raid/nfs_storageIPB/thesis_JL/copy_of_thesisdata/interim/DEM/"
}
fobs = "PV_energy_comprehensive_615W_Mono_id_day_19850101-20201231.nc"
fDEM = "DEM_regrided_0p1deg_Indonesia.nc"
VARS = ['rsds', 'clt', 'tas', 'hurs', 'sfcWind']  # climate variables
EXPERIMENTS = ['historical']  # we will work with 2 future scenarios later, SSP2-4.5 & SSP5-8.5
modelsname = ['MPI-ESM1-2-HR']  # list of CMIP6 models that available, let's put 1 model for now

# Define area of interest (part of Java Island)
lon_bnds, lat_bnds = (104, 115), (-9, -5)

# Preprocess input data
partial_func = partial(prep._preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

# Define helper functions
def get_sorted_files(var, model, experiment):
    """Get sorted list of files for a given variable, model and experiment."""
    pattern = f"{PATHS['CMIP6']}{var}/{model}/{var}_day_{model}_{experiment}_*.nc"
    return sorted(glob(pattern))

def get_all_experiment_files():
    """Get all experiment files for all variables in a single dictionary."""
    all_files = {}
    for var in VARS:
        all_files[var] = {
            exp: get_sorted_files(var, modelsname[0], exp)
            for exp in EXPERIMENTS
        }
    return all_files

# Get all files
all_files = get_all_experiment_files()

# Filter files by year ranges
filtered_files = prep.filter_all_files_by_year_ranges(
    all_files,
    exps=EXPERIMENTS,
    baseline_range=(1985, 2014), 
    scenario_range=(2015, 2100)
)

# Open all datasets
datasets = read.open_all_datasets(filtered_files, VARS, partial_func)

# Access specific data
baseline_rsds_ds = datasets['baseline']['rsds']
baseline_sfcWind_ds = datasets['baseline']['sfcWind']
baseline_tas_ds = datasets['baseline']['tas']
baseline_clt_ds = datasets['baseline']['clt']
baseline_hurs_ds = datasets['baseline']['hurs']

#scenario_tas_ds = results['datasets']['scenario']['tas']

# Read reference data for target
PV_ds = xr.open_dataset(PATHS['OBS']+fobs)
pv_output = PV_ds.sel(**{'longitude': slice(*lon_bnds),
                             'latitude': slice(lat_bnds[1], lat_bnds[0]), # Reverse for descending order
                             'valid_time': slice('1985','2014')}
                         )
# Read elevation data
DEM_ds = xr.open_dataset(PATHS['DEM']+fDEM)
sel_DEM_ds = DEM_ds.sel(**{'longitude': slice(*lon_bnds),
                             'latitude': slice(lat_bnds[1], lat_bnds[0])}
                         )
# Create dictionary of data arrays and convert from dask array data array
climate_data_dict = {
    'sfcWind': baseline_sfcWind_ds.drop_vars('height').compute(), # drop 1-length coordinate variable
    'tas': baseline_tas_ds.drop_vars('height').compute(),     
    'hurs': baseline_hurs_ds.drop_vars('height').compute(),          
    'rsds': baseline_rsds_ds.compute(),        
    'clt': baseline_clt_ds.compute()
}

# Rename dimension of elevation for consistency
sel_DEM_renamed = sel_DEM_ds.rename({
    'latitude': 'lat', 
    'longitude': 'lon'}
                                   )
sel_DEM_da = sel_DEM_renamed.elevation.squeeze(dim='band', drop=True)

# Convert calendar from model output if it doesn't equal to standard/gregorian
# Get length of sample historical period for reference data
obs_length = len(pv_output.valid_time.values)

# Baseline
new_climate_data_dict = {}
for var_name, ds in climate_data_dict.items():
    
    if var_name in VARS: 
        model_length = len(ds[var_name].time.values)
        
        if model_length != obs_length: # proceed conversion
            print(f"The calendar is not standard. Starting calendar conversion for {var_name}...")
            
            # Convert calendar to standard/proleptic gregorian
            std_ds = ds.convert_calendar(
                "proleptic_gregorian", dim='time', align_on="year", missing=np.nan)
        
            # Interpolate missing value between dates after converting calendar
            std_ds = std_ds.interpolate_na(dim="time")
            new_climate_data_dict[var_name] = std_ds 
        else:
            new_climate_data_dict[var_name] = ds 
            print("The calendar system is standard")


import xesmf as xe

# Rename reference data dimension for consistency
pv_rename_dims = pv_output.rename({
    'valid_time': 'time',
    'latitude': 'lat', 
    'longitude': 'lon'
})

# Define the target lat-lon grid based on observation data
ds_targetgrid = xr.Dataset(
    {
        "lat": (("lat"), pv_rename_dims.lat.values), # Y
        "lon": (("lon"), pv_rename_dims.lon.values), # X
    }
)

def regrid_spatial(ds, target_grid, method="nearest_s2d"):
    """Perform spatial interpolation to make data resolution consistent ."""
    regridder = xe.Regridder(
        ds,
        target_grid,
        method,
    )
    return regridder(ds)

# Perform regrid_spatial for climate data dictionary
regridd_climate_dict = {}

for var_name in new_climate_data_dict:
    ds = new_climate_data_dict[var_name]
    regridd_climate_dict[var_name] = regrid_spatial(ds, ds_targetgrid)

# Get days of the year (J)
day_of_year = pv_rename_dims.time.dt.dayofyear

# Using broadcast_like
number_of_day_da = day_of_year.broadcast_like(pv_rename_dims)
lat_da = pv_rename_dims.lat.broadcast_like(pv_rename_dims)
lon_da = pv_rename_dims.lon.broadcast_like(pv_rename_dims)
DEM_da = sel_DEM_da.broadcast_like(pv_rename_dims)

# Create mask ocean array
mask_ocean = (pv_rename_dims.PV_basic_per_m2[0,:,:] > 0)

# Create helper function to apply land-sea mask
def ocean_mask(da):
    """Use land-sea mask to eliminate values in ocean grid points"""
    # Returns elements from da where condition is True, and fill in locations where False with NA
    masked_da = da * mask_ocean
    
    return masked_da.where(masked_da != 0) 

lat_masked = ocean_mask(lat_da)
lon_masked = ocean_mask(lon_da)
day_masked = ocean_mask(number_of_day_da)
DEM_masked = ocean_mask(DEM_da)
climate_data_masked = {}

for var_name in regridd_climate_dict:
    ds = regridd_climate_dict[var_name]
    climate_data_masked[var_name] = ocean_mask(ds[var_name])

# Standardize all data across time dimension
def preprocessing_data(data_dict, pv_output, lat_da, lon_da, dayofyear_da, DEM_da):
    """
    Standardize each grid cell independently across time.
    """
    
    # Inisialize new dict to store standardize data
    data_dict_scaled = {}

    # 1. Climate predictor
    for var_name,da in data_dict.items():
        print(f"Processing {var_name}")

        # Calculate mean and std of all grid across time
        da = data_dict[var_name]
        global_mean = da.mean()
        global_std = da.std()
        
        # Standardize: (x - mean) / std
        da_scaled = (da - global_mean) / global_std
        data_dict_scaled[var_name] = da_scaled
        
    # 2. Latitude (constant in time, varies by grid)
    lat_mean = lat_da.mean()
    lat_std = lat_da.std()
    lat_scaled = (lat_da - lat_mean) / lat_std
    data_dict_scaled['lat'] = lat_scaled
    print(f"Processing latitude")

    # 3. Longitude (constant in time, varies by grid)
    lon_mean = lon_da.mean()
    lon_std = lon_da.std()
    lon_scaled = (lon_da - lon_mean) / lon_std
    data_dict_scaled['lon'] = lon_scaled
    print(f"Processing longitude")

    # 4. Day of year. It is cyclical. Only use sine and cosine encoding (no scaling)
    days_in_year = 365.25
    doy_rad = 2 * np.pi * dayofyear_da / days_in_year # Direct encoding (xarray handles broadcasting automatically)
    doy_sin = np.sin(doy_rad)
    doy_cos = np.cos(doy_rad)
    data_dict_scaled['doy_sin'] = doy_sin
    data_dict_scaled['doy_cos'] = doy_cos
    print(f"Processing day of year")
    
    # 5. Elevation (constant in time, varies by grid)
    DEM_mean = DEM_da.mean()
    DEM_std = DEM_da.std()
    DEM_scaled = (DEM_da - DEM_mean) / DEM_std
    data_dict_scaled['elevation'] = DEM_scaled
    print(f"Processing elevation")
    
    # 6. PV target - WITH NaN HANDLING
    pv_da = pv_output.PV_basic_per_m2
    pv_mean = pv_da.mean()
    pv_std = pv_da.std()
    pv_scaled = (pv_da - pv_mean) / pv_std
    data_dict_scaled['pv_target'] = pv_scaled
    print(f"Processing PV (target)")
    
    return data_dict_scaled

historical_data_scaled = preprocessing_data(
    climate_data_masked, pv_rename_dims, lat_masked, lon_masked, day_masked, DEM_masked)

def create_ann_input_vectorized(scaled_dict, climate_vars):
    """
    Vectorized version for preparing ann input.
    """
    # Get climate variables as arrays
    climate_arrays = []
    for var in climate_vars:
        # Shape: (time, lat, lon)
        climate_arrays.append(scaled_dict[var].values)
    
    # Stack climate variables
    # Each becomes shape: (time, lat, lon)
    # We want: (time, lat, lon, 5) for 5 climate vars
    climate_stack = np.stack(climate_arrays, axis=-1)  # (time, lat, lon, 5)
    
    # Get other features
    lat_array = scaled_dict['lat'].values  # (lat, lon) or (time, lat, lon)
    lon_array = scaled_dict['lon'].values  # (lat, lon) or (time, lat, lon)
    doy_sin_array = scaled_dict['doy_sin'].values  # (time, lat, lon) or (time,)
    doy_cos_array = scaled_dict['doy_cos'].values  # (time, lat, lon) or (time,)
    elevation_array = scaled_dict['elevation'].values
    pv_array = scaled_dict['pv_target'].values  # (time, lat, lon)
    
    # Get dimensions
    n_time, n_lat, n_lon, n_climate = climate_stack.shape
    n_grid = n_lat * n_lon
    
    # Flatten spatial dimensions
    climate_flat = climate_stack.reshape(n_time, n_grid, n_climate)  # (time, grid, 5)
    
    # Flatten lat/lon
    lat_flat = lat_array.reshape(n_time, n_grid) # (time, lat, lon)
    lon_flat = lon_array.reshape(n_time, n_grid) # (time, lat, lon)
    
    # Flatten day sine and cosine
    doy_sin_flat = doy_sin_array.reshape(n_time, n_grid) # (time, lat, lon)
    doy_cos_flat = doy_cos_array.reshape(n_time, n_grid)

    # Flatten elevation
    elevation_flat = elevation_array.reshape(n_time, n_grid)
    
    # Combine all features
    # (time, grid, 10) where 10 = 5 climate + lat + lon + day_sine + day_cosine + elevation
    X_3d = np.zeros((n_time, n_grid, 10))
    X_3d[:, :, :5] = climate_flat  # Climate variables
    X_3d[:, :, 5] = lat_flat       # Latitude
    X_3d[:, :, 6] = lon_flat       # Longitude
    X_3d[:, :, 7] = doy_sin_flat   # Day sine
    X_3d[:, :, 8] = doy_cos_flat   # Day cosine
    X_3d[:, :, 9] = elevation_flat # Elevation
    
    # Flatten time and grid dimensions
    X_2d = X_3d.reshape(-1, 10)  # (time*grid, 10)
    
    # Flatten PV target
    y_2d = pv_array.reshape(-1)  # (time*grid,)
    
    # Remove samples with NaN in EITHER X OR y
    print("Checking for NaN values:")
    print(f"  NaN in y: {np.isnan(y_2d).sum():,} samples")
    print(f"  NaN in X: {np.isnan(X_2d).any(axis=1).sum():,} samples")
    
    # Create mask for valid samples (no NaN anywhere)
    valid_mask_X = ~np.isnan(X_2d).any(axis=1)
    valid_mask_y = ~np.isnan(y_2d)
    valid_mask = valid_mask_X & valid_mask_y
    
    X_clean = X_2d[valid_mask]
    y_clean = y_2d[valid_mask]
    
    print(f"\nTotal possible samples: {X_2d.shape[0]:,}")
    print(f"Samples with NaN in features only: {np.sum(~valid_mask_X & valid_mask_y):,}")
    print(f"Samples with NaN in target only: {np.sum(valid_mask_X & ~valid_mask_y):,}")
    print(f"Samples with NaN in both: {np.sum(~valid_mask_X & ~valid_mask_y):,}")
    print(f"Valid samples (no NaN anywhere): {X_clean.shape[0]:,}")
    
    # Final sanity check
    print(f"\nSanity check after cleaning:")
    print(f"  NaN in X_clean: {np.isnan(X_clean).any()}")
    print(f"  NaN in y_clean: {np.isnan(y_clean).any()}")
    
    # At the end, return X_clean, y_clean, valid_mask
    return X_clean, y_clean, valid_mask

X, y, valid_mask = create_ann_input_vectorized(
    historical_data_scaled, 
    VARS
)

print(f"\nFinal ANN dataset:")
print(f"X shape: {X.shape}  (samples × features)")
print(f"y shape: {y.shape}  (samples)")
print(f"Features per sample: {X.shape[1]}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Check setup
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensor datasets WITHOUT moving to device
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),  # Remove .to(device)
    torch.FloatTensor(y_train)   # Remove .to(device)
)

test_dataset = TensorDataset(
    torch.FloatTensor(X_test),   # Remove .to(device)
    torch.FloatTensor(y_test)    # Remove .to(device)
)

# Create DataLoaders with optimized settings
batch_size = 3000  # Adjust based on your GPU memory
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=2,  # Parallel data loading
    pin_memory=True,  # Set to True for faster CPU to GPU transfer
    prefetch_factor=2  # Prefetch batches
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model defenition
class PVDownscaling(nn.Module):
    def __init__(self, in_features=10, h1=4096, h2=2048, h3=1024, out_features=1, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(h3, out_features)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
        
# Initialize model and move to device
torch.manual_seed(41)
model = PVDownscaling().to(device)  

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', # Monitor validation loss (for loss, we want it to decrease. So,lower is better)
    factor=0.5, 
    patience=5 # Give it more time to learn, reduce LR after 5 epochs without improvement
)

# ===========================================
# EARLY STOPPING SETUP
# ===========================================
epochs = 500
patience = 15  # Stop after 15 epochs without improvement
min_delta = 1e-5  # Minimum improvement threshold
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
best_model_state = None

# ===========================================
# SPECIFIED OUTPUT PATH FOR BEST MODEL
# ===========================================
import os
save_dir = '/raid/nfs_storageIPB/thesis_JL/modelling/'

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'best_model_4096_2048_1024_edit.pth')

train_losses = []
val_losses = []

print(f"Starting training for max {epochs} epochs...")
print(f"Early stopping patience: {patience}, min_delta: {min_delta}")
print(f"Best model will be saved to: {save_path}")

# Verify directory is writable
if os.access(save_dir, os.W_OK):
    print(f"✅ Directory is writable: {save_dir}")
else:
    print(f"⚠️ Warning: Directory may not be writable: {save_dir}")

for epoch in range(epochs):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    total_train_loss = 0
    num_train_batches = 0
    
    # Add progress bar for training batches
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
    for batch_X, batch_y in train_pbar:
        # Move batch to GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        y_pred = model(batch_X)
        loss = criterion(y_pred.squeeze(), batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        num_train_batches += 1
        
        # Update progress bar with current loss
        train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_train_loss = total_train_loss / num_train_batches
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    
    val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
    with torch.no_grad():
        for batch_X, batch_y in val_pbar:
            # Move batch to GPU
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            val_pred = model(batch_X)
            val_loss = criterion(val_pred.squeeze(), batch_y)
            total_val_loss += val_loss.item()
            num_val_batches += 1
            
            val_pbar.set_postfix({'loss': f'{val_loss.item():.6f}'})
    
    avg_val_loss = total_val_loss / num_val_batches
    
    # Store losses
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']
    
    # ===========================================
    # EARLY STOPPING CHECK (Run this BEFORE scheduler)
    # ===========================================
    if avg_val_loss < best_val_loss - min_delta:
        # Improvement found
        improvement = best_val_loss - avg_val_loss
        print(f'Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, '
              f'Val Loss = {avg_val_loss:.6f} ✅ IMPROVED by {improvement:.6f} '
              f'({best_val_loss:.6f} → {avg_val_loss:.6f}), LR = {current_lr:.2e}, '
              f'Time = {epoch_time:.1f}s')
        
        best_val_loss = avg_val_loss
        best_epoch = epoch
        patience_counter = 0
        
        # Save best model to specified path
        best_model_state = model.state_dict().copy()
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'architecture': '4096→2048→1024',
            'dropout': 0.30,
            'lr': current_lr,
            'input_features': 10,
            'batch_size': batch_size,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        }, save_path)
        print(f"  💾 Best model saved to {save_path}")
    else:
        # No improvement
        patience_counter += 1
        print(f'Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, '
              f'Val Loss = {avg_val_loss:.6f} ⚠ No improvement ({patience_counter}/{patience}), '
              f'LR = {current_lr:.2e}, Time = {epoch_time:.1f}s')
    
    # Update learning rate scheduler (AFTER early stopping check)
    scheduler.step(avg_val_loss)
    
    # Check if we should stop
    if patience_counter >= patience:
        print(f"\n{'='*60}")
        print(f"🛑 EARLY STOPPING TRIGGERED after {epoch+1} epochs!")
        print(f"Best model from epoch {best_epoch+1} with validation loss: {best_val_loss:.6f}")
        print(f"{'='*60}")
        break

# ===========================================
# LOAD BEST MODEL AND EVALUATE
# ===========================================
print(f"\nLoading best model from epoch {best_epoch+1}...")
if best_model_state is not None:
    model.load_state_dict(best_model_state)
else:
    print("Warning: No best model saved, using current model")

# ===========================================
# COMPREHENSIVE METRICS CALCULATION (TRAIN & TEST)
# ===========================================
from scipy import stats

def calculate_metrics(model, data_loader, device, dataset_name="Dataset"):
    """
    Calculate comprehensive metrics for a given data loader
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # Calculate metrics
    r2 = r2_score(all_targets, all_preds)
    #mae = mean_absolute_error(all_targets, all_preds)
    #rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    #mse = mean_squared_error(all_targets, all_preds)
    
    # Correlation coefficient
    correlation_coefficient, p_value = stats.pearsonr(all_targets, all_preds)
    r_from_r2 = np.sqrt(abs(r2)) * (1 if correlation_coefficient > 0 else -1)
    
    # Additional statistics
    residuals = all_targets - all_preds
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    return {
        'dataset': dataset_name,
        'r2': r2,
        'correlation': correlation_coefficient,
        'p_value': p_value,
        'r_from_r2': r_from_r2,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'predictions': all_preds,
        'targets': all_targets,
        'residuals': residuals
    }

# Calculate metrics for both train and test sets
print(f"\n{'='*60}")
print(f"CALCULATING METRICS FOR TRAIN AND TEST SETS")
print(f"{'='*60}")

train_metrics = calculate_metrics(model, train_loader, device, "TRAINING_SET")
test_metrics = calculate_metrics(model, test_loader, device, "TEST_SET")

# ===========================================
# DISPLAY COMPREHENSIVE RESULTS
# ===========================================
print(f"\n{'='*70}")
print(f"COMPREHENSIVE MODEL PERFORMANCE (Best Model from Epoch {best_epoch+1})")
print(f"{'='*70}")

# Header
print(f"\n{'Metric':<25} {'TRAINING SET':<20} {'TEST SET':<20} {'DIFFERENCE':<15}")
print(f"{'-'*80}")

# Rows
print(f"{'R² Score':<25} {train_metrics['r2']:<20.4f} {test_metrics['r2']:<20.4f} "
      f"{(train_metrics['r2'] - test_metrics['r2']):<+15.4f}")

print(f"{'Correlation (R)':<25} {train_metrics['correlation']:<20.4f} {test_metrics['correlation']:<20.4f} "
      f"{(train_metrics['correlation'] - test_metrics['correlation']):<+15.4f}")

print(f"{'R from √R²':<25} {train_metrics['r_from_r2']:<20.4f} {test_metrics['r_from_r2']:<20.4f} "
      f"{(train_metrics['r_from_r2'] - test_metrics['r_from_r2']):<+15.4f}")

print(f"{'P-value':<25} {train_metrics['p_value']:<20.4e} {test_metrics['p_value']:<20.4e} "
      f"{'':<15}")

print(f"{'Residual Mean':<25} {train_metrics['residual_mean']:<20.4f} {test_metrics['residual_mean']:<20.4f} "
      f"{(train_metrics['residual_mean'] - test_metrics['residual_mean']):<+15.4f}")

print(f"{'Residual Std Dev':<25} {train_metrics['residual_std']:<20.4f} {test_metrics['residual_std']:<20.4f} "
      f"{(train_metrics['residual_std'] - test_metrics['residual_std']):<+15.4f}")

print(f"{'='*80}")

# ===========================================
# SAVE METRICS TO FILE
# ===========================================
metrics_path = os.path.join(save_dir, 'comprehensive_metrics_4096_2048_1024_edit.txt')
with open(metrics_path, 'w') as f:
    f.write(f"COMPREHENSIVE MODEL PERFORMANCE (Best Model from Epoch {best_epoch+1})\n")
    f.write(f"{'='*80}\n\n")
    f.write(f"Architecture: 3072→1536→768\n")
    f.write(f"Dropout Rate: 0.3\n")
    f.write(f"Best Epoch: {best_epoch+1}\n")
    f.write(f"Early Stopping at: {epoch+1} epochs\n\n")
    f.write(f"{'Metric':<25} {'TRAINING SET':<20} {'TEST SET':<20}\n")
    f.write(f"{'-'*65}\n")
    f.write(f"{'R² Score':<25} {train_metrics['r2']:<20.4f} {test_metrics['r2']:<20.4f}\n")
    f.write(f"{'Correlation (R)':<25} {train_metrics['correlation']:<20.4f} {test_metrics['correlation']:<20.4f}\n")
    f.write(f"{'P-value':<25} {train_metrics['p_value']:<20.4e} {test_metrics['p_value']:<20.4e}\n")
    f.write(f"{'Residual Mean':<25} {train_metrics['residual_mean']:<20.4f} {test_metrics['residual_mean']:<20.4f}\n")
    f.write(f"{'Residual Std':<25} {train_metrics['residual_std']:<20.4f} {test_metrics['residual_std']:<20.4f}\n")
    f.write(f"{'='*65}\n")
print(f"\n✅ Comprehensive metrics saved to: {metrics_path}")
