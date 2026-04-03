import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

def make_datafiles(params, sev):
    # 1. Determine data paths based on severity (sev)
    if sev == 1:
        data_path = params.get('nompath', params.get('nomflightpath'))
    elif sev == 2:
        data_path = params.get('s2flightpath', params.get('valpath'))
    else:
        data_path = params.get('s3flightpath', params.get('testpath'))

    # 2. Load .mat files
    mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    raw_data_list = []
    
    for file in mat_files:
        # loadmat loads MATLAB files into a Python dictionary
        mat_contents = loadmat(os.path.join(data_path, file))
        # Assuming the data is stored under a key 'm' in the mat file
        if 'm' in mat_contents:
            raw_data_list.append(mat_contents['m'])

    # Combine all flight data vertically
    m_full = np.vstack(raw_data_list)
    
    # 3. Extract Continuous Channels and Target
    continuous_channels = params.get('channelContineous', [])
    target_name = params.get('targetName', 'target')
    
    # Assuming we have a way to map target_name to a column index
    target_idx = params.get('targetIdx', -1) 
    input_indices = [i for i in range(len(continuous_channels)) if i != target_idx]
    
    X_raw = m_full[:, input_indices]
    y_raw = m_full[:, target_idx]

    # 4. Z-Score Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Save Raw and Scaled Data to CSV
    raw_df = pd.DataFrame(np.column_stack((X_raw, y_raw)))
    raw_df.to_csv(f"{params['datapath']}/rawtrainingdata.csv", index=False)
    
    scaled_df = pd.DataFrame(np.column_stack((X_scaled, y_raw)))
    scaled_df.to_csv(f"{params['datapath']}/zscorenormalizedtrainingdata.csv", index=False)

    # 5. Feature Correlation Extraction
    # Calculate correlation matrix using Pandas
    corr_matrix = scaled_df.corr().abs()
    
    # Extract correlation of inputs against the target (last column)
    target_correlations = corr_matrix.iloc[:-1, -1]
    
    # Sort features by highest correlation
    sorted_features = target_correlations.sort_values(ascending=False)
    
    features = {
        'list': sorted_features.index.tolist(),
        'corr': sorted_features.values.tolist()
    }

    # 6. Support Vector Regression (SVR) Optimization 
    # This completely replaces the manual MATLAB createJob parallel grid search
    if 'svr' in params.get('algo', []) and sev == 2:
        print("Running preliminary optimization for SVR...")
        
        # Define hyperparameter grid
        param_grid = {
            'C': loguniform(1e-2, 1e4).rvs(10), # Random samples between 0.01 and 10000
            'epsilon': loguniform(1e-4, 1e1).rvs(10),
            'gamma': loguniform(1e-4, 1e1).rvs(10) # Equivalent to sigma
        }
        
        svr = SVR(kernel='rbf')
        
        # RandomizedSearchCV handles the parallelization automatically using n_jobs=-1
        random_search = RandomizedSearchCV(
            svr, param_distributions=param_grid, n_iter=20, 
            scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1
        )
        
        random_search.fit(X_scaled, y_raw)
        
        print(f"Best SVR Parameters found: {random_search.best_params_}")
        params['C'] = random_search.best_params_['C']
        params['epsilon'] = random_search.best_params_['epsilon']
        params['sigma'] = random_search.best_params_['gamma']

    # 7. Package and save to .mat for downstream compatibility
    output_dict = {
        'train': {'x': X_scaled, 'y': y_raw},
        'features': features,
        'Statistics': {'dataMean': scaler.mean_, 'dataStd': scaler.scale_}
    }
    
    savemat(f"{params['datapath']}/processed_data_sev_{sev}.mat", output_dict)

    return output_dict