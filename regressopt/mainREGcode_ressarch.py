import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

class Struct:
    """Lightweight class for dot-notation."""
    pass

def mainREGcode_ressarch(x: float, tr, tst, algo_list: list, runOptions) -> tuple:
    """
    Builds data-driven models using various regression algorithms via scikit-learn.
    
    Parameters:
    - x: The hyperparameter tuning value (meaning changes based on algorithm)
    - tr: Training data object (tr.x, tr.y)
    - tst: Test data object containing lists of batches (tst.x[ik], tst.y[ik])
    - algo_list: List of algorithm strings to run
    - runOptions: Configuration and model caching object
    """
    output = Struct()
    
    # Default to 1 test batch if not specified
    Ntests = getattr(runOptions, 'Ntests', len(tst.y) if hasattr(tst, 'y') else 1)
    
    algo_names = ['gp', 'svr', 'libsvr', 'knn', 'btree', 'lin', 'quad', 'bnet', 'elm', 'ransac']
    
    # Find matching algorithm indices
    D = [algo_names.index(a) + 1 for a in algo_list if a in algo_names]
    
    # Initialize output prediction arrays
    output.yhat = [None] * Ntests
    for a in algo_list:
        setattr(output, f"{a}_yhat", [None] * Ntests)

    # Loop through each test data batch
    for ik in range(Ntests):
        # Extract the current batch of test data
        tst_x_batch = tst.x[ik]
        tst_y_batch = tst.y[ik]
        
        for algo_idx in D:
            
            # 1. Gaussian Process
            if algo_idx == 1:
                if not hasattr(runOptions, 'modelGP'):
                    # Using default RBF kernel for GP; 'x' could scale the kernel bounds
                    model = GaussianProcessRegressor(random_state=0)
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.gpTrainTime = time.time() - t0
                    runOptions.modelGP = model
                else:
                    model = runOptions.modelGP
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.gpTestTime = time.time() - t0

            # 2. Support Vector Regression (SVR)
            elif algo_idx == 2:
                if not hasattr(runOptions, 'modelSVR'):
                    # 'x' maps to C (regularization) or gamma. Let's map it to C.
                    c_val = x if x > 0 else 1e-4
                    model = make_pipeline(StandardScaler(), SVR(C=c_val, kernel='rbf'))
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.svrTrainTime = time.time() - t0
                    runOptions.modelSVR = model
                else:
                    model = runOptions.modelSVR
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.svrTestTime = time.time() - t0

            # 3. LibSVR (Linear SVR)
            elif algo_idx == 3:
                if not hasattr(runOptions, 'modelLibSVR'):
                    model = make_pipeline(StandardScaler(), LinearSVR(C=x, random_state=0))
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.lsvmTrainTime = time.time() - t0
                    runOptions.modelLibSVR = model
                else:
                    model = runOptions.modelLibSVR
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.lsvmTestTime = time.time() - t0

            # 4. K-Nearest Neighbors
            elif algo_idx == 4:
                # x is number of neighbors
                k_val = max(1, int(round(x)))
                k_val = min(k_val, len(tr.y))  # Cannot have more neighbors than samples
                
                model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k_val))
                t0 = time.time()
                model.fit(tr.x, tr.y)
                output.kNNTrainTime = time.time() - t0
                
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.kNNTestTime = time.time() - t0

            # 5. Bagged Trees (Random Forest)
            elif algo_idx == 5:
                if not hasattr(runOptions, 'modelbtree'):
                    # x acts as min_samples_leaf
                    min_leaf = max(1, int(round(x)))
                    model = RandomForestRegressor(min_samples_leaf=min_leaf, random_state=0)
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.bagTreeTrainTime = time.time() - t0
                    runOptions.modelbtree = model
                else:
                    model = runOptions.modelbtree
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.bagTreeTestTime = time.time() - t0

            # 6. Linear Regression (Regularized Ridge)
            elif algo_idx == 6:
                if not hasattr(runOptions, 'modelRsumLin'):
                    # x acts as lambda (alpha in sklearn Ridge)
                    model = make_pipeline(StandardScaler(), Ridge(alpha=x))
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.RsumLinTrainTime = time.time() - t0
                    runOptions.modelRsumLin = model
                else:
                    model = runOptions.modelRsumLin
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.RsumLinTestTime = time.time() - t0

            # 7. Quadratic Regression
            elif algo_idx == 7:
                if not hasattr(runOptions, 'modelQuad'):
                    # Use a pipeline to generate polynomial features, then fit Ridge
                    model = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=x))
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.QuadTrainTime = time.time() - t0
                    runOptions.modelQuad = model
                else:
                    model = runOptions.modelQuad
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.QuadTestTime = time.time() - t0

            # 8. Bagged Neural Networks
            elif algo_idx == 8:
                if not hasattr(runOptions, 'modelbnets'):
                    # x is number of hidden units
                    hidden_units = max(1, int(round(x)))
                    base_nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(hidden_units,), max_iter=500))
                    model = BaggingRegressor(estimator=base_nn, n_estimators=10, random_state=0)
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.bagTrainTime = time.time() - t0
                    runOptions.modelbnets = model
                else:
                    model = runOptions.modelbnets
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.bagTestTime = time.time() - t0

            # 9. Extreme Learning Machine (ELM)
            elif algo_idx == 9:
                # Scikit-learn does not have native ELMs. 
                # Falling back to a standard MLP/Neural Net with x nodes.
                if not hasattr(runOptions, 'modelELM'):
                    nodes = max(1, int(round(x)))
                    model = MLPRegressor(hidden_layer_sizes=(nodes,), max_iter=1000)
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.elmTrainTime = time.time() - t0
                    runOptions.modelELM = model
                else:
                    model = runOptions.modelELM
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.elmTestTime = time.time() - t0

            # 10. RANSAC (Robust Regression)
            elif algo_idx == 10:
                if not hasattr(runOptions, 'modelRANSAC'):
                    # x represents residual threshold
                    model = RANSACRegressor(residual_threshold=x, random_state=0)
                    t0 = time.time()
                    model.fit(tr.x, tr.y)
                    output.ransacTrainTime = time.time() - t0
                    runOptions.modelRANSAC = model
                else:
                    model = runOptions.modelRANSAC
                    
                t0 = time.time()
                y_pred = model.predict(tst_x_batch)
                output.ransacTestTime = time.time() - t0
            
            # Store predictions
            # If multiple algorithms are run simultaneously, save to specific array
            algo_name = algo_names[algo_idx - 1]
            if len(D) > 1:
                getattr(output, f"{algo_name}_yhat")[ik] = y_pred
            else:
                output.yhat[ik] = y_pred

    return output, runOptions