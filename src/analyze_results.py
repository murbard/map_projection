import optuna

def analyze():
    study = optuna.load_study(study_name='map_projection_study_v3', storage='sqlite:///study_results.db')
    
    print("-" * 50)
    print(f"Study: {study.study_name}")
    print(f"Total Trials: {len(study.trials)}")
    print(f"Best Value: {study.best_value}")
    print(f"Best Params: {study.best_params}")
    print("-" * 50)
    
    completed_trials = [t for t in study.trials if t.value is not None]
    best_trials = sorted(completed_trials, key=lambda t: t.value)[:10]
    
    print(f"{'Loss':<10} | {'L':<5} | {'H':<5} | {'Batch':<8} | {'Act':<10} | {'LR':<10}")
    print("-" * 60)
    
    for t in best_trials:
        p = t.params
        print(f"{t.value:<10.4f} | {p['n_layers']:<5} | {p['hidden_dim']:<5} | {p['batch_size']:<8} | {p['activation']:<10} | {p['lr']:.5f}")

if __name__ == "__main__":
    analyze()
