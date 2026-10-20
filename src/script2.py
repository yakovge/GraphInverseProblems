import os
from script import create_message, run_commands_sequentially

# Leave-one-dataset-out: train on all 5 tasks over every snapshot of the training datasets,
# then test on all 5 tasks over the whole held-out dataset.
DATASETS = ['CPOX', 'PEDALME', 'WIKIMATHS', 'MONTEVIDEO', 'WINDMILL']
TEST_ONLY = ['WINDMILL', 'WIKIMATHS']  # only ever tested: Windmill is a complete graph (101k edges); a 4-graph WikiMaths batch
                                       # (220k edges) needs > 14 GB of activations and 8-15 s per step on CPU
TEST_FRAC = {'WINDMILL': 0.05, 'WIKIMATHS': 0.2}  # 17k / 731 snapshots: a fraction of the test series is enough

dic = {
    'device': os.environ.get('GRIP_DEVICE', 'cuda:0'),  # GRIP_DEVICE=cpu runs on CPU (slow; see GPU_RUN.md)
    'dataset': 'MULTI',
    'epochs': 40,
    'head_epochs': 16,
    'test_head_epochs': 4,
    'dc': 'prox',  # data consistency: 'prox' = learned per-task step, 'cgls' = original fixed CGLS steps
    'channels': 32,   # 32 channels x 32 layers = ~101k parameters (supervisor's 100-200k brief); 128 x 32 was 1.58M
    'num_seeds': int(os.environ.get('GRIP_SEEDS', 3)),   # seeds 0..k-1 -> checkpoints <name>_seed_k.pth; tables report mean ± std
    'layers': 32,
    'heads': True,  # False: no task-specific heads -> head_epochs/test_head_epochs are skipped
    'noise': True,
    'painting': True,
    'blurring': True,
    'sensoring': True,
    'pdessm': True,
    'method': 'foundation',
    'cglsIter': 5,
    'solveIter': 5,
    'classify': 0,
    'max_patience': 10000,
    'mask_per_snapshot_budget': 6,
}

if __name__ == "__main__":
    my_commands = []
    for test in DATASETS:
        dic['train_datasets'] = ','.join(d for d in DATASETS if d != test and d not in TEST_ONLY)
        dic['test_dataset'] = test
        dic['test_frac'] = TEST_FRAC.get(test, 1.0)
        dic['project_name'] = f'epoch_40_head_16_test_4_{test.lower()}_dataset_test'
        my_commands.append(create_message(dic))
    run_commands_sequentially(my_commands)
