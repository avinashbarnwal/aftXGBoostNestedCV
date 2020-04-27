import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import optuna
import time
import json
import argparse
from multiprocessing import cpu_count

from optuna.samplers import RandomSampler

def data_import(data_name):
    filename = 'https://raw.githubusercontent.com/avinashbarnwal/GSOC-2019/master/AFT/test/data/'+data_name+'/'
    inputFileName = filename + 'inputs.csv'
    labelFileName = filename + 'outputs.csv'
    foldsFileName = filename + 'cv/equal_labels/folds.csv'
    inputs        = pd.read_csv(inputFileName, index_col='sequenceID')
    labels        = pd.read_csv(labelFileName, index_col='sequenceID')
    folds         = pd.read_csv(foldsFileName, index_col='sequenceID')
    res           = {}
    res['inputs'] = inputs
    res['labels'] = labels
    res['folds']  = folds
    return(res)

def preprocess_data(inputs,labels):
    inputs.replace([-float('inf'), float('inf')], np.nan, inplace=True)
    missingCols = inputs.isnull().sum()
    missingCols = list(missingCols[missingCols > 0].index)
    print(f'missingCols = {missingCols}')

    inputs.drop(missingCols,axis=1,inplace=True)
    
    varCols     = inputs.apply(lambda x: np.var(x))
    zeroVarCols = list(varCols[varCols==0].index)
    print(f'zeroVarCols = {zeroVarCols}')
    
    inputs.drop(zeroVarCols,axis=1,inplace=True)
    labels['min.lambda'] = labels['min.log.lambda'].apply(lambda x: np.exp(x))
    labels['max.lambda'] = labels['max.log.lambda'].apply(lambda x: np.exp(x))
    
    return inputs, labels

def get_train_valid_test_splits(folds, test_fold_id, inputs, labels, kfold_gen):
    # Split data into train and test
    X            = inputs[folds['fold'] != test_fold_id].values
    X_test       = inputs[folds['fold'] == test_fold_id].values
    y_label      = labels[folds['fold'] != test_fold_id]
    y_label_test = labels[folds['fold'] == test_fold_id]
    
    dtest = xgb.DMatrix(X_test)
    dtest.set_float_info('label_lower_bound', y_label_test['min.lambda'].values)
    dtest.set_float_info('label_upper_bound', y_label_test['max.lambda'].values)
    
    # Further split train into train and valid. Do this 5 times to obtain 5 fold cross validation
    folds = []
    dmat_train_valid_combined = xgb.DMatrix(X)
    dmat_train_valid_combined.set_float_info('label_lower_bound', y_label['min.lambda'].values)
    dmat_train_valid_combined.set_float_info('label_upper_bound', y_label['max.lambda'].values)
    for train_idx, valid_idx in kfold_gen.split(X):
        dtrain = xgb.DMatrix(X[train_idx, :])
        dtrain.set_float_info('label_lower_bound', y_label['min.lambda'].values[train_idx])
        dtrain.set_float_info('label_upper_bound', y_label['max.lambda'].values[train_idx])
        
        dvalid = xgb.DMatrix(X[valid_idx, :])
        dvalid.set_float_info('label_lower_bound', y_label['min.lambda'].values[valid_idx])
        dvalid.set_float_info('label_upper_bound', y_label['max.lambda'].values[valid_idx])
        
        folds.append((dtrain, dvalid))
    
    return (folds, dmat_train_valid_combined, dtest)

base_params = {'verbosity': 0,
               'objective': 'survival:aft',
               'tree_method': 'hist',
               'nthread': 1,
               'eval_metric': 'interval-regression-accuracy'}  # Hyperparameters common to all trials

def accuracy(predt, dmat):
    y_lower = dmat.get_float_info('label_lower_bound')
    y_upper = dmat.get_float_info('label_upper_bound')
    acc = np.sum((predt >= y_lower) & (predt <= y_upper)) / len(predt)
    return 'accuracy', acc

def train(trial, train_valid_folds, dtest, distribution):
    eta              = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
    max_depth        = trial.suggest_int('max_depth', 2, 10, step=2)
    min_child_weight = trial.suggest_loguniform('min_child_weight',0.1, 100.0)
    reg_alpha        = trial.suggest_loguniform('reg_alpha', 0.0001, 100)
    reg_lambda       = trial.suggest_loguniform('reg_lambda', 0.0001, 100)
    sigma            = trial.suggest_loguniform('aft_loss_distribution_scale', 1.0, 100.0)
    
    params = {'eta': eta,
              'max_depth': int(max_depth),
              'min_child_weight': min_child_weight,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'aft_loss_distribution': distribution,
              'aft_loss_distribution_scale': sigma}
    params.update(base_params)
    
    # Cross validation metric is computed as follows:
    # 1. For each of the 5 folds, run XGBoost for 5000 rounds and record trace of the validation metric (accuracy)
    # 2. Compute the mean validation metric over the 5 folds, for each iteration ID.
    # 3. Select the iteration ID which maximizes the mean validation metric.
    # 4. Return the mean validation metric as CV metric.
    validation_metric_history = pd.DataFrame()
    for fold_id, (dtrain, dvalid) in enumerate(train_valid_folds):
        res = {}
        bst = xgb.train(params, dtrain, num_boost_round=5000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        verbose_eval=False, evals_result=res)
        validation_metric_history[fold_id] = res['valid']['interval-regression-accuracy']
    validation_metric_history['mean'] = validation_metric_history.mean(axis=1)
    best_num_round = validation_metric_history['mean'].idxmax()

    trial.set_user_attr('num_round', best_num_round)
    trial.set_user_attr('timestamp', time.time())

    return validation_metric_history.iloc[best_num_round].mean()

def run_nested_cv(inputs, labels, folds, seed, dataset_name):
    fold_ids = np.unique(folds['fold'].values)

    for distribution in ['normal', 'logistic', 'extreme']:
        # Nested Cross-Validation, with 4-folds CV in the outer loop and 5-folds CV in the inner loop
        for test_fold_id in fold_ids:
            start = time.time()
            # train_valid_folds: list of form [(train_set, valid_set), ...], where train_set is used for training
            #                    and valid_set is used for model selection, i.e. hyperparameter search
            # dtest: held-out test set; will not be used for training or model selection
            kfold_gen = KFold(n_splits=5, shuffle=True, random_state=seed)
            train_valid_folds, dtrain_valid_combined, dtest \
              = get_train_valid_test_splits(folds, test_fold_id, inputs, labels, kfold_gen)
            
            study = optuna.create_study(sampler=RandomSampler(seed=seed), direction='maximize')
            study.optimize(lambda trial : train(trial, train_valid_folds, dtest, distribution), n_trials=100,
                           n_jobs=cpu_count() // 2)
            
            # Use the best hyperparameter set to fit a model with all data points except the held-out test set
            best_params = study.best_params
            best_num_round = study.best_trial.user_attrs['num_round']
            best_params.update(base_params)
            best_params['aft_loss_distribution'] = distribution
            final_model = xgb.train(best_params, dtrain_valid_combined,
                                    num_boost_round=best_num_round,
                                    evals=[(dtrain_valid_combined, 'train-valid'), (dtest, 'test')],
                                    verbose_eval=False)
            
            # Evaluate accuracy on the test set
            # Accuracy = % of data points for which the final model produces a prediction that falls within the label range
            acc = accuracy(final_model.predict(dtest), dtest)
            print(f'Fold {test_fold_id}: Accuracy {acc}')
            final_model.save_model(f'{dataset_name}/{distribution}-fold{test_fold_id}-model.json')

            with open(f'{dataset_name}/{distribution}-fold{test_fold_id}.json', 'w') as f:
                trials = study.get_trials(deepcopy=False)
                trial_id = [trial.number for trial in trials]
                score = [trial.value for trial in trials]
                timestamp = [trial.user_attrs['timestamp'] - start for trial in trials]
                json.dump({'trial_id': trial_id, 'cv_accuracy': score, 'timestamp': timestamp, 'final_accuracy': acc}, f)
            
        end = time.time()    
        time_taken = end - start
        print(f'Time elapsed = {time_taken}, distribution = {distribution}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', required=False, type=int, default=1)

    args = parser.parse_args()
    print(f'Dataset = {args.dataset}')
    print(f'Using {cpu_count() // 2} threads to run hyperparameter search')
    data  = data_import(args.dataset)
    inputs = data['inputs']
    labels = data['labels']
    folds  = data['folds']

    inputs, labels = preprocess_data(inputs, labels)
    run_nested_cv(inputs, labels, folds, seed=args.seed, dataset_name=args.dataset)

if __name__ == '__main__':
    main()
