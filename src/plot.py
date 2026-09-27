import torch
import matplotlib.pyplot as plt
import os

from utils import process_data, get_data_and_loaders, get_network, get_forward_op
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
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith(".pth"):
                filepath = os.path.join(model_dir, filename)
                
                # Create a dummy forward operation to initialize the network architecture
                dummy_op = get_forward_op(args, args.channels, label_channels, device)
                
                # Instantiate the model architecture
                net = get_network(args, dummy_op, args.channels, label_channels, feat_channels, device)
                
                # Load the saved state dict
                state_dict = torch.load(filepath, map_location=device, weights_only=True)
                net.load_state_dict(state_dict)
                net.eval()
                
                loaded_models.append({
                    "name": filename,
                    "model": net
                })
                print(f" -> Successfully loaded model weights from {filename}")
    else:
        print(f"Directory '{model_dir}' not found. No models loaded.")
        
    return loaded_models

def run_operations_and_models(models, operations, dataset_loader, args, device):
    """
    Takes the loaded models, operations, and test dataset.
    Applies each operation to the data, runs the models on the transformed data, 
    and calculates the reconstruction loss, returning it as a matrix (dictionary).
    """
    print("Running operations and calculating losses on the dataset...")
    
    # Initialize loss matrix: model_name -> op_name -> total_loss
    loss_matrix = {model['name']: {op_name: 0.0 for op_name in operations.keys()} for model in models}
    batch_counts = {model['name']: {op_name: 0 for op_name in operations.keys()} for model in models}
    
    with torch.no_grad():
        for graph in dataset_loader:
            graph = process_data(args, graph)
            graph = graph.to(device)
            
            for op_name, op_instance in operations.items():
                # Apply forward operation
                forward_data = op_instance(graph.y, graph.edge_index, graph.edge_weight, emb=False)
                
                # Run each model on the transformed data
                for model_info in models:
                    net = model_info['model']
                    
                    # Ensure the model knows which task it's evaluating if it's the foundation model
                    if args.method == 'foundation' and hasattr(net, 'set_task'):
                        task_mapping = {
                            'noise': 'denoising',
                            'painting': 'inpainting',
                            'blurring': 'source_localization',
                            'sensoring': 'sensor_recovery',
                            'pdessm': 'pde_reconstruction'
                        }
                        net.set_task(task_mapping.get(op_name, op_name))
                        
                    # Model inference
                    if args.method in ['laplacian_regularization', 'tikhonov_regularization', 'laplacian_explicit']:
                        X = net(forward_data, graph)
                    else:
                        X, Xref, R = net(forward_data, graph.edge_index, graph.edge_weight, graph.x)
                        
                    # Calculate relative MSE loss (loss_X)
                    loss_X = F.mse_loss(X, graph.y) / F.mse_loss(torch.zeros_like(graph.y), graph.y)
                    
                    loss_matrix[model_info['name']][op_name] += loss_X.item()
                    batch_counts[model_info['name']][op_name] += 1
    
    # Average the losses across all batches
    for model_name in loss_matrix:
        for op_name in loss_matrix[model_name]:
            if batch_counts[model_name][op_name] > 0:
                loss_matrix[model_name][op_name] /= batch_counts[model_name][op_name]
                
    return loss_matrix

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

    # Define arguments to mimic main_3_linear_inv_problems.py perfectly
    args = argparse.Namespace(
        dataset='CPOX',
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
        noise=False,
        painting=False,
        blurring=False,
        sensoring=False,
        pdessm=False
    )

    print("Loading CPOX Dataset via identical main script loader...")
    train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels = get_data_and_loaders(args)
    
    # We need a sample to determine dimensions to prevent out-of-bounds errors
    sample = next(iter(test_dataset))
    num_nodes = sample.x.shape[0]
    args.num_nodes = num_nodes
    loaded_models = load_all_models(args, label_channels, feat_channels, device)

    print(f"Total models ready for evaluation: {len(loaded_models)}")
    
    # Create a generic tensor of node indices for masking and sensoring tasks.
    dummy_indices = torch.arange(num_nodes // 2, device=device) 
    
    operations = load_foundation_operations(
        nin=label_channels,
        embdsize=args.channels,
        dummy_indices=dummy_indices,
        device=device
    )

    print("\nSuccessfully loaded operations.")
    
    # Run operations and evaluate models to get the loss matrix
    loss_matrix = run_operations_and_models(
        loaded_models, operations, test_loader, args, device
    )

    # Plot the matrix and save as a PNG
    plot_loss_matrix_table(loss_matrix)

    print("\nReady for plotting pipeline integration.")

if __name__ == "__main__":
    main()