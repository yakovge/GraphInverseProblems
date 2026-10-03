from script import create_message, run_commands_sequentially

# Leave-one-dataset-out: train on all 5 tasks over every snapshot of the training datasets,
# then test on all 5 tasks over the whole held-out dataset.
DATASETS = ['CPOX', 'PEDALME', 'WIKIMATHS', 'MONTEVIDEO', 'WINDMILL']
TEST_ONLY = ['WINDMILL']  # complete graph (101k edges): a training step does not fit a 6 GB GPU, so it is only ever tested

dic = {
    'dataset': 'MULTI',
    'epochs': 40,
    'head_epochs': 16,
    'test_head_epochs': 4,
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
        dic['test_frac'] = 0.05 if test == 'WINDMILL' else 1.0  # Windmill has 17k snapshots
        dic['project_name'] = f'epoch_40_head_16_test_4_{test.lower()}_dataset_test'
        my_commands.append(create_message(dic))
    run_commands_sequentially(my_commands)
