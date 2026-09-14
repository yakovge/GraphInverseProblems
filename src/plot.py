import torch
import matplotlib.pyplot as plt
import os

from utils import process_data, get_data_and_loaders, get_network, get_forward_op
from utils import (load_checkpoint, load_legacy_state_dict, evaluate_all_operators,
                   eval_args_from_checkpoint, build_loss_matrix, generate_measurement,
                   sample_operator_config, apply_config, compute_metric,
                   FLAG_TO_TASK, ALL_FLAGS)
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F

# Importing the 5 forward operations from the local graphForwardOps module
from graphForwardOps import (
    AddNoise,
    graphMask,
    graph_smooth,
    SensorRecovery,
    PDESSM
)
import argparse
from utils import process_data

def load_foundation_operations(nin, embdsize, dummy_indices, device, noise_std=0.1, blur_k=4, pdessm_dim=32):
    """
    Initializes and returns a dictionary containing the 5 foundation model operations.
    Accepts basic configuration parameters for noise, painting, blurring, sensoring, and pdessm.
    """
    print("Loading Foundation Model Operations...")
    
    noise_op = AddNoise(nin=nin, embdsize=embdsize, noise_std=noise_std, device=device, learnEmb=True)
    painting_op = graphMask(ind=dummy_indices, embdsize=embdsize, nin=nin, device=device, learnEmb=True)
    blurring_op = graph_smooth(nin=nin, embdsize=embdsize, k=blur_k, device=device, learnEmb=True)
    sensoring_op = SensorRecovery(sensor_indices=dummy_indices, nin=nin, embdsize=embdsize, device=device, learnEmb=True)
    pdessm_op = PDESSM(nin=nin, embdsize=embdsize, dim=pdessm_dim, device=device, learnEmb=True)

    operations = {
        'noise': noise_op,
        'painting': painting_op,
        'blurring': blurring_op,
        'sensoring': sensoring_op,
        'pdessm': pdessm_op
    }
    
    return operations

def load_all_models(args, label_channels, feat_channels, device, model_dir="models"):
    """
    Scans the specified directory for .pth files and loads them as PyTorch models.
    """
    print(f"Dynamically loading saved models from '{model_dir}' directory...")
    loaded_models = []
    
    from utils import build_foundation_model_from_checkpoint
    if os.path.exists(model_dir):
        for filename in sorted(os.listdir(model_dir)):
            if filename.endswith(".pth"):
                filepath = os.path.join(model_dir, filename)

                # Read metadata FIRST and reconstruct the model from the checkpoint's
                # own architecture (channels, backbone_type, ...), so a nondefault
                # checkpoint (e.g. channels-8) loads regardless of caller defaults.
                blob = torch.load(filepath, map_location=device, weights_only=False)
                meta = {}
                if isinstance(blob, dict) and 'model_state_dict' in blob:
                    net, meta = build_foundation_model_from_checkpoint(filepath, feat_channels, device)
                else:
                    # Plain (legacy) state_dict -> build with caller args + migrate.
                    net = get_network(args, None, args.channels, label_channels, feat_channels, device)
                    load_legacy_state_dict(filepath, net, device)
                net.eval()

                loaded_models.append({"name": filename, "model": net, "meta": meta})
                print(f" -> Successfully loaded model weights from {filename}")
    else:
        print(f"Directory '{model_dir}' not found. No models loaded.")

    return loaded_models


def run_operations_and_models(models, args, device, loader_factory=None):
    """Build the loss matrix using the production evaluation protocol.

    Delegates to utils.build_loss_matrix, which groups models by their COMPLETE
    saved protocol and builds a loader per group (so different dataset/batch size/
    splits are scored on matching batching), emitting per-group baselines with
    unique ids. `loader_factory(margs)->test_loader` defaults to the production
    foundation loader; tests inject a synthetic factory.
    """
    print("Running production evaluation for each model...")
    return build_loss_matrix(models, args, device, loader_factory=loader_factory,
                             process_fn=process_data, op_names=ALL_FLAGS)

def plot_loss_matrix_table(loss_matrix, save_path="loss_matrix_table.png"):
    """
    Takes the computed loss matrix and generates a nicely formatted 
    matplotlib table, saving it as an image file.
    """
    print(f"Generating visual table for the loss matrix...")
    
    model_names = list(loss_matrix.keys())
    if not model_names:
        print("Loss matrix is empty, nothing to plot.")
        return

    op_names = list(loss_matrix[model_names[0]].keys())
    
    # Prepare data for the table
    cell_text = []
    for model in model_names:
        row = [f"{loss_matrix[model][op]:.4f}" for op in op_names]
        cell_text.append(row)
        
    # Dynamically size the figure based on rows and cols
    fig_width = max(8, len(op_names) * 1.5 + 4)
    fig_height = max(3, len(model_names) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ax.axis('off')
    ax.axis('tight')
    
    # Define colors for a nice aesthetic
    header_color = '#40466e'
    row_colors = ['#f1f1f2', '#ffffff'] * ((len(model_names) + 1) // 2)
    
    table = ax.table(cellText=cell_text,
                     rowLabels=model_names,
                     colLabels=op_names,
                     loc='center',
                     cellLoc='center',
                     rowColours=['#e9ecef']*len(model_names),
                     colColours=[header_color]*len(op_names))
                     
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8) # Stretch width and height
    
    # Adjust colors and fonts for header and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
        elif col == -1: # Row labels
            cell.set_text_props(weight='bold')
        elif row > 0:
            cell.set_facecolor(row_colors[(row - 1) % len(row_colors)])
            
    plt.title("Reconstruction Loss (Relative MSE) by Operation", pad=20, weight='bold', size=14)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Table successfully saved to {save_path}")

def main():
    """
    Main execution function for plot.py.
    Loads the CPOX dataset identically to main script, dynamically loads saved 
    PyTorch models, and creates a dictionary of transformed datasets.
    """
    print("Initializing environment...")
    
    # Setup device (defaults to cuda if available, otherwise cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Arguments matching the foundation training/eval protocol.
    args = argparse.Namespace(
        dataset='CPOX',
        datapath='./data',
        use_meta_data=1,
        classify=0,
        CPOX_lags=1,
        train_batch_size=4,
        test_batch_size=4,
        train_frac=1.0,
        test_frac=1.0,
        method='foundation',
        layers=16,
        channels=32,
        cglsIter=5,
        solveIter=5,
        rnfPE=1,
        dropout=0.0,
        task='mask',
        blur_count='4',
        mask_per_snapshot_budget=16,
        held_out_op=None,
        noise=True, painting=True, blurring=True, sensoring=True, pdessm=True,
    )

    # Loaders are built PER PROTOCOL inside build_loss_matrix (dataset/splits/batch
    # size reproduced from each checkpoint's saved eval_config).
    loaded_models = load_all_models(args, 1, 1, device)
    print(f"Total models ready for evaluation: {len(loaded_models)}")
    loss_matrix = run_operations_and_models(loaded_models, args, device)

    # Plot the matrix and save as a PNG
    plot_loss_matrix_table(loss_matrix)

    print("\nReady for plotting pipeline integration.")

if __name__ == "__main__":
    main()