"""
XAU/USD Deep Learning Price Prediction
Ultra-Deep Neural Network with Billions of Parameters
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import warnings
import datetime
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("XAU/USD ULTRA-DEEP NEURAL NETWORK PREDICTOR")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*80)

# ============================================================================
# 1. FETCH EXTENSIVE HISTORICAL DATA
# ============================================================================
print("\n[1/10] Fetching extensive historical data...")

def fetch_multi_timeframe_data():
    """Fetch data from multiple sources and timeframes"""
    
    # Main gold futures data
    tickers = {
        'GC=F': 'Gold Futures',
        'GLD': 'Gold ETF',
        'DX-Y.NYB': 'Dollar Index',
        '^TNX': '10-Year Treasury',
        '^VIX': 'VIX',
        'CL=F': 'Crude Oil',
        'SI=F': 'Silver Futures',
        '^GSPC': 'S&P 500',
        'BTC-USD': 'Bitcoin'
    }
    
    # Fetch maximum available history
    start_date = "2010-01-01"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    all_data = {}
    for ticker, name in tickers.items():
        print(f"  - Downloading {name} ({ticker})...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                all_data[ticker] = data
                print(f"    ✓ Got {len(data)} records")
            else:
                print(f"    ✗ No data available")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    return all_data

data_dict = fetch_multi_timeframe_data()
main_data = data_dict['GC=F']  # Gold futures as main target

# Fix multi-level columns if present
if isinstance(main_data.columns, pd.MultiIndex):
    main_data.columns = main_data.columns.get_level_values(0)

# Ensure all data has single-level columns
for key in data_dict:
    if isinstance(data_dict[key].columns, pd.MultiIndex):
        data_dict[key].columns = data_dict[key].columns.get_level_values(0)

print(f"\nTotal data points collected: {len(main_data)}")
print(f"Date range: {main_data.index[0]} to {main_data.index[-1]}")

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] Creating advanced features...")

def create_advanced_features(df, data_dict):
    """Create extensive feature set"""
    
    result = df.copy()
    
    # Ensure we're working with single-level columns
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    
    # Basic price features
    result['Returns'] = result['Close'].pct_change()
    result['Log_Returns'] = np.log(result['Close'] / result['Close'].shift(1))
    result['Price_Change'] = result['Close'].diff()
    
    # Safe division with replace
    result['High_Low_Ratio'] = result['High'] / result['Low'].replace(0, np.nan)
    result['Close_Open_Ratio'] = result['Close'] / result['Open'].replace(0, np.nan)
    
    # Volume features
    result['Volume_Change'] = result['Volume'].pct_change()
    vol_ma = result['Volume'].rolling(20).mean().replace(0, np.nan)
    result['Volume_MA_Ratio'] = result['Volume'] / vol_ma
    
    # Multiple timeframe moving averages
    windows = [5, 10, 20, 50, 100, 200]
    for window in windows:
        sma = result['Close'].rolling(window).mean()
        ema = result['Close'].ewm(span=window, adjust=False).mean()
        result[f'SMA_{window}'] = sma
        result[f'EMA_{window}'] = ema
        result[f'Price_to_SMA_{window}'] = result['Close'] / sma
    
    # Bollinger Bands
    for window in [10, 20, 50]:
        sma = result['Close'].rolling(window).mean()
        std = result['Close'].rolling(window).std()
        result[f'BB_upper_{window}'] = sma + (std * 2)
        result[f'BB_lower_{window}'] = sma - (std * 2)
        
        # Safe division
        sma_safe = sma.replace(0, np.nan)
        bb_range = result[f'BB_upper_{window}'] - result[f'BB_lower_{window}']
        bb_range_safe = bb_range.replace(0, np.nan)
        
        result[f'BB_width_{window}'] = bb_range / sma_safe
        result[f'BB_position_{window}'] = (result['Close'] - result[f'BB_lower_{window}']) / bb_range_safe
    
    # Volatility measures
    for window in [5, 10, 20, 50]:
        result[f'Volatility_{window}'] = result['Returns'].rolling(window).std()
        result[f'ATR_{window}'] = (result['High'] - result['Low']).rolling(window).mean()
    
    # RSI (Relative Strength Index)
    for period in [7, 14, 21, 28]:
        delta = result['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        
        # Safe division - replace 0 loss with very small number to avoid inf
        loss_safe = loss.replace(0, 1e-10)
        rs = gain / loss_safe
        result[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = result['Close'].ewm(span=12, adjust=False).mean()
    exp2 = result['Close'].ewm(span=26, adjust=False).mean()
    result['MACD'] = exp1 - exp2
    result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
    
    # Stochastic Oscillator
    for period in [14, 21]:
        low_min = result['Low'].rolling(period).min()
        high_max = result['High'].rolling(period).max()
        result[f'Stoch_{period}'] = 100 * (result['Close'] - low_min) / (high_max - low_min)
        result[f'Stoch_{period}_MA'] = result[f'Stoch_{period}'].rolling(3).mean()
    
    # Momentum indicators
    for period in [5, 10, 20]:
        result[f'Momentum_{period}'] = result['Close'] - result['Close'].shift(period)
        
        # Safe ROC calculation
        close_shifted = result['Close'].shift(period).replace(0, np.nan)
        result[f'ROC_{period}'] = ((result['Close'] - result['Close'].shift(period)) / close_shifted) * 100
    
    # Price patterns (lagged features)
    for lag in range(1, 31):  # 30 days of history
        result[f'Close_Lag_{lag}'] = result['Close'].shift(lag)
        result[f'Return_Lag_{lag}'] = result['Returns'].shift(lag)
    
    # Seasonal features
    result['Day_of_Week'] = result.index.dayofweek
    result['Month'] = result.index.month
    result['Quarter'] = result.index.quarter
    result['Day_of_Month'] = result.index.day
    try:
        result['Week_of_Year'] = result.index.isocalendar().week.astype(int)
    except:
        result['Week_of_Year'] = result.index.week
    
    # Cyclical encoding of time features
    result['Day_of_Week_Sin'] = np.sin(2 * np.pi * result['Day_of_Week'] / 7)
    result['Day_of_Week_Cos'] = np.cos(2 * np.pi * result['Day_of_Week'] / 7)
    result['Month_Sin'] = np.sin(2 * np.pi * result['Month'] / 12)
    result['Month_Cos'] = np.cos(2 * np.pi * result['Month'] / 12)
    
    # Add correlated assets
    for ticker, data in data_dict.items():
        if ticker != 'GC=F' and not data.empty:
            # Align dates
            aligned = data.reindex(result.index, method='ffill')
            prefix = ticker.replace('=', '_').replace('^', '').replace('-', '_').replace('.', '_')
            if 'Close' in aligned.columns:
                result[f'{prefix}_Close'] = aligned['Close']
                result[f'{prefix}_Returns'] = aligned['Close'].pct_change()
                result[f'{prefix}_SMA_20'] = aligned['Close'].rolling(20).mean()
    
    # Replace any remaining inf values
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # Statistical features
    for window in [5, 10, 20]:
        result[f'Skew_{window}'] = result['Returns'].rolling(window).skew()
        result[f'Kurt_{window}'] = result['Returns'].rolling(window).kurt()
    
    return result

df_features = create_advanced_features(main_data, data_dict)

# Drop NaN values
df_features = df_features.dropna()

print(f"Total features created: {len(df_features.columns)}")
print(f"Data points after feature engineering: {len(df_features)}")

# ============================================================================
# 3. PREPARE DATA FOR ULTRA-DEEP NETWORK
# ============================================================================
print("\n[3/10] Preparing data for deep learning...")

# Select all numerical features except target
feature_cols = [col for col in df_features.columns if col not in ['Close', 'Adj Close']]
X = df_features[feature_cols].values

# Target: predict next day's closing price
y = df_features['Close'].shift(-1).values

# Remove last row (no target)
X = X[:-1]
y = y[:-1]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Clean data - handle inf and extreme values
print("Cleaning data...")
# Replace inf with NaN
X = np.where(np.isinf(X), np.nan, X)
y = np.where(np.isinf(y), np.nan, y)

# Check for any remaining issues
print(f"  - NaN values in X: {np.isnan(X).sum()}")
print(f"  - Inf values in X: {np.isinf(X).sum()}")

# Replace NaN with column median
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Remove any rows where target is still NaN
valid_idx = ~np.isnan(y)
X = X[valid_idx]
y = y[valid_idx]

print(f"  - Data points after cleaning: {len(X)}")

# Clip extreme outliers (beyond 10 standard deviations)
for i in range(X.shape[1]):
    col_mean = np.mean(X[:, i])
    col_std = np.std(X[:, i])
    if col_std > 0:
        lower_bound = col_mean - 10 * col_std
        upper_bound = col_mean + 10 * col_std
        X[:, i] = np.clip(X[:, i], lower_bound, upper_bound)

print("Data cleaning complete!")

# Train-test split (80-20, chronological)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scale features using RobustScaler (better for outliers)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print("Data scaling complete!")

# ============================================================================
# 4. BUILD ULTRA-DEEP NEURAL NETWORK (BILLIONS OF PARAMETERS)
# ============================================================================
print("\n[4/10] Building ultra-deep neural network...")

def build_ultra_deep_model(input_dim):
    """
    Build a massive deep neural network with billions of parameters
    """
    
    model = keras.Sequential(name='XAU_USD_Ultra_Deep_Network')
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # First block: Massive expansion
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    # Second block: Deep feature extraction
    model.add(layers.Dense(3072, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Dense(3072, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_4'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    # Third block: Complex pattern recognition
    model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_5'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_6'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_7'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    # Fourth block: Refinement layers
    model.add(layers.Dense(1536, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_8'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(1536, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_9'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_10'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_11'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.15))
    
    # Fifth block: Deep abstraction
    model.add(layers.Dense(768, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_12'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.15))
    
    model.add(layers.Dense(768, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_13'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.15))
    
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_14'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_15'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    
    # Sixth block: Final refinement
    model.add(layers.Dense(384, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_16'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_17'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='dense_18'))
    model.add(layers.Dense(64, activation='relu', name='dense_19'))
    model.add(layers.Dense(32, activation='relu', name='dense_20'))
    
    # Output layer
    model.add(layers.Dense(1, name='output'))
    
    return model

model = build_ultra_deep_model(X_train_scaled.shape[1])

# Compile with advanced optimizer
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model.compile(
    optimizer=optimizer,
    loss='huber',  # More robust to outliers than MSE
    metrics=['mae', 'mse']
)

print("\n" + "="*80)
print("MODEL ARCHITECTURE SUMMARY")
print("="*80)
model.summary()

# Count parameters
total_params = model.count_params()
print("\n" + "="*80)
print(f"TOTAL PARAMETERS: {total_params:,}")
print(f"THAT'S {total_params/1e9:.3f} BILLION PARAMETERS!")
print("="*80)

# ============================================================================
# 5. VISUALIZE NETWORK ARCHITECTURE
# ============================================================================
print("\n[5/10] Visualizing network architecture...")

# Plot model architecture
try:
    keras.utils.plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=150
    )
    print("✓ Model architecture diagram saved as 'model_architecture.png'")
except Exception as e:
    print(f"✗ Could not create architecture diagram: {e}")

# Create custom visualization
def visualize_network_structure(model):
    """Create custom network visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Plot 1: Layer sizes
    layer_names = []
    layer_sizes = []
    layer_types = []
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            layer_names.append(f"L{i+1}")
            layer_sizes.append(layer.units)
            layer_types.append(type(layer).__name__)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_sizes)))
    bars = ax1.barh(layer_names, layer_sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Number of Neurons', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Layer', fontsize=14, fontweight='bold')
    ax1.set_title('Network Layer Sizes', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, size) in enumerate(zip(bars, layer_sizes)):
        ax1.text(size + 50, i, f'{size:,}', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Parameter distribution
    param_counts = []
    param_labels = []
    
    for i, layer in enumerate(model.layers):
        params = layer.count_params()
        if params > 0:
            param_counts.append(params)
            param_labels.append(f"L{i+1}")
    
    ax2.pie(param_counts[:10], labels=param_labels[:10], autopct='%1.1f%%', 
            startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, 10)))
    ax2.set_title('Parameter Distribution (Top 10 Layers)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Network visualization saved as 'network_visualization.png'")
    plt.close()

visualize_network_structure(model)

# ============================================================================
# 6. SETUP TRAINING CALLBACKS
# ============================================================================
print("\n[6/10] Setting up training callbacks...")

# Create directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint(
    'checkpoints/model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=1e-7,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True
)

print("✓ Callbacks configured")

# ============================================================================
# 7. TRAIN THE ULTRA-DEEP NETWORK
# ============================================================================
print("\n[7/10] Training ultra-deep neural network...")
print("="*80)
print("TRAINING CONFIGURATION:")
print(f"  - Epochs: 500 (with early stopping)")
print(f"  - Batch size: 64")
print(f"  - Validation split: 20%")
print(f"  - Optimizer: Adam with adaptive learning rate")
print(f"  - Loss function: Huber (robust to outliers)")
print("="*80)

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard],
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# 8. VISUALIZE TRAINING HISTORY
# ============================================================================
print("\n[8/10] Visualizing training history...")

def plot_training_history(history):
    """Create comprehensive training visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2, alpha=0.8)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (Huber)', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # MAE plot
    axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2, alpha=0.8)
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # MSE plot
    axes[1, 0].plot(history.history['mse'], label='Training MSE', linewidth=2, alpha=0.8)
    axes[1, 0].plot(history.history['val_mse'], label='Validation MSE', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MSE', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='red', alpha=0.8)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        # Training vs Validation comparison
        epochs = len(history.history['loss'])
        train_val_ratio = [t/v if v != 0 else 1 for t, v in zip(history.history['loss'], history.history['val_loss'])]
        axes[1, 1].plot(train_val_ratio, linewidth=2, color='purple', alpha=0.8)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', label='Perfect fit')
        axes[1, 1].set_title('Training/Validation Loss Ratio', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Loss / Val Loss', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history saved as 'training_history.png'")
    plt.close()

plot_training_history(history)

# ============================================================================
# 9. EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n[9/10] Evaluating model performance...")

# Make predictions
y_pred_train_scaled = model.predict(X_train_scaled, verbose=0)
y_pred_test_scaled = model.predict(X_test_scaled, verbose=0)

# Inverse transform
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred.flatten()) / y_true)) * 100
    
    print(f"\n{dataset_name} METRICS:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return mae, rmse, r2, mape

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)

train_metrics = calculate_metrics(y_train, y_pred_train, "TRAINING SET")
test_metrics = calculate_metrics(y_test, y_pred_test, "TEST SET")

print("="*80)

# ============================================================================
# 10. CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("\n[10/10] Creating comprehensive visualizations...")

def create_prediction_visualizations(y_train, y_pred_train, y_test, y_pred_test):
    """Create detailed prediction visualizations"""
    
    # Figure 1: Time series predictions
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    # Training predictions
    axes[0].plot(range(len(y_train)), y_train, label='Actual Price', linewidth=2, alpha=0.7, color='blue')
    axes[0].plot(range(len(y_pred_train)), y_pred_train, label='Predicted Price', linewidth=2, alpha=0.7, color='red')
    axes[0].set_title('Training Set: Actual vs Predicted Prices', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Price (USD)', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Test predictions
    axes[1].plot(range(len(y_test)), y_test, label='Actual Price', linewidth=2, alpha=0.7, color='blue')
    axes[1].plot(range(len(y_pred_test)), y_pred_test, label='Predicted Price', linewidth=2, alpha=0.7, color='red')
    axes[1].set_title('Test Set: Actual vs Predicted Prices', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Price (USD)', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_timeseries.png', dpi=300, bbox_inches='tight')
    print("✓ Time series predictions saved as 'predictions_timeseries.png'")
    plt.close()
    
    # Figure 2: Scatter plots and error analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # Training scatter
    axes[0, 0].scatter(y_train, y_pred_train, alpha=0.5, s=10)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_title('Training: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Actual Price (USD)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Price (USD)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test scatter
    axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, s=10, color='orange')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_title('Test: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Price (USD)', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Price (USD)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training errors
    train_errors = y_train - y_pred_train.flatten()
    axes[1, 0].hist(train_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Training Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Prediction Error (USD)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test errors
    test_errors = y_test - y_pred_test.flatten()
    axes[1, 1].hist(test_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Test Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Prediction Error (USD)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction analysis saved as 'predictions_analysis.png'")
    plt.close()
    
    # Figure 3: Detailed error analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Error over time (training)
    axes[0, 0].plot(train_errors, linewidth=1, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].fill_between(range(len(train_errors)), train_errors, 0, alpha=0.3)
    axes[0, 0].set_title('Training Errors Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time Steps', fontsize=12)
    axes[0, 0].set_ylabel('Error (USD)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error over time (test)
    axes[0, 1].plot(test_errors, linewidth=1, alpha=0.6, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].fill_between(range(len(test_errors)), test_errors, 0, alpha=0.3, color='orange')
    axes[0, 1].set_title('Test Errors Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time Steps', fontsize=12)
    axes[0, 1].set_ylabel('Error (USD)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Percentage error (training)
    train_pct_errors = (train_errors / y_train) * 100
    axes[1, 0].plot(train_pct_errors, linewidth=1, alpha=0.6, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Training Percentage Errors', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time Steps', fontsize=12)
    axes[1, 0].set_ylabel('Error (%)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Percentage error (test)
    test_pct_errors = (test_errors / y_test) * 100
    axes[1, 1].plot(test_pct_errors, linewidth=1, alpha=0.6, color='purple')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Test Percentage Errors', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Time Steps', fontsize=12)
    axes[1, 1].set_ylabel('Error (%)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Error analysis saved as 'error_analysis.png'")
    plt.close()

create_prediction_visualizations(y_train, y_pred_train, y_test, y_pred_test)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE - SUMMARY")
print("="*80)
print(f"Model: Ultra-Deep Neural Network")
print(f"Total Parameters: {model.count_params():,} ({model.count_params()/1e9:.3f} billion)")
print(f"Total Layers: {len(model.layers)}")
print(f"Training Samples: {len(X_train):,}")
print(f"Test Samples: {len(X_test):,}")
print(f"Features Used: {X.shape[1]}")
print(f"\nTest Set Performance:")
print(f"  MAE:  ${test_metrics[0]:.2f}")
print(f"  RMSE: ${test_metrics[1]:.2f}")
print(f"  R²:   {test_metrics[2]:.6f}")
print(f"  MAPE: {test_metrics[3]:.2f}%")
print("\nOutput Files:")
print("  ✓ model_architecture.png - Network architecture diagram")
print("  ✓ network_visualization.png - Layer sizes and parameter distribution")
print("  ✓ training_history.png - Training metrics over epochs")
print("  ✓ predictions_timeseries.png - Actual vs predicted prices")
print("  ✓ predictions_analysis.png - Scatter plots and error distributions")
print("  ✓ error_analysis.png - Detailed error analysis")
print("  ✓ checkpoints/model_best.h5 - Best model weights")
print("  ✓ logs/ - TensorBoard logs")
print("="*80)

# Make a future prediction
last_features = X_test_scaled[-1:]
next_prediction_scaled = model.predict(last_features, verbose=0)
next_prediction = scaler_y.inverse_transform(next_prediction_scaled)

print(f"\nNEXT DAY PREDICTION:")
print(f"  Current Price: ${y_test[-1]:.2f}")
print(f"  Predicted Price: ${next_prediction[0][0]:.2f}")
print(f"  Expected Change: ${next_prediction[0][0] - y_test[-1]:.2f} ({((next_prediction[0][0] - y_test[-1]) / y_test[-1] * 100):.2f}%)")
print("="*80)

print("\nTo view training progress in TensorBoard, run:")
print("  tensorboard --logdir=logs")
print("\n🚀 Happy trading! 🚀")
