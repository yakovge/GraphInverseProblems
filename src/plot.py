import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import process_data, task_specific_modifiers

def compare_operators_and_save_table(net, train_ops, test_ops, train_loader, test_loader, args):
    """
    Evaluates a neural network against 5 different operators (for both train and test),
    generates a colored comparison table, and saves it as a JPEG in the 'models/' directory.
    
    Args:
        net: The neural network model.
        train_ops (list): A list of 5 forward operators to use during train data evaluation.
        test_ops (list): A list of 5 forward operators to use during test data evaluation.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
        args: Arguments namespace (must contain project_name, method, classify, device, etc.)
    """
    device = args.device

    # Helper function to evaluate the network on a specific loader and operator[cite: 1]
    def evaluate(loader, op, dataset):
        net.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        
        # Handle cases where ops are passed as a list [operator_object, op_name][cite: 1]
        op_name = op[1] if isinstance(op, list) else args.task
        actual_op = op[0] if isinstance(op, list) else op
        
        # Switch the foundation model to the current task if applicable[cite: 1]
        if hasattr(net, 'set_task'):
            net.set_task(op_name)
            
        for graph in loader:
            with torch.no_grad():
                # Extract batch size safely[cite: 1]
                actual_bs = len(graph.batch.unique()) if hasattr(graph, 'batch') else 1
                total_batches += actual_bs
                
                # Apply data preprocessing and modifiers[cite: 1]
                graph = process_data(args, graph)
                graph, actual_op = task_specific_modifiers(graph, args, actual_op, dataset)
                graph = graph.to(device)
                
                # Generate forward data[cite: 1]
                forward_data = actual_op(graph.y, graph.edge_index, graph.edge_weight, emb=False)
                
                # Model inference[cite: 1]
                if args.method in ['laplacian_regularization', 'tikhonov_regularization', 'laplacian_explicit']:
                    X = net(forward_data, graph)
                else:
                    X, Xref, R = net(forward_data, graph.edge_index, graph.edge_weight, graph.x)
                    
                # Reconstruct forward data[cite: 1]
                forward_data_rec = actual_op(X, graph.edge_index, graph.edge_weight, emb=False)
                
                # Loss and Accuracy Calculation[cite: 1]
                if args.classify:
                    loss_X = F.cross_entropy(X, graph.y)
                    loss_data = F.cross_entropy(forward_data_rec, forward_data)
                    loss = loss_X + loss_data
                    
                    pred = torch.argmax(X, dim=-1)
                    acc = torch.eq(pred, torch.argmax(graph.y, dim=-1)).sum() / graph.x.shape[0]
                else: # Regression
                    loss_X = F.mse_loss(X, graph.y) / (F.mse_loss(torch.zeros_like(graph.y), graph.y) + 1e-8)
                    loss_data = F.mse_loss(forward_data_rec, forward_data) / (F.mse_loss(torch.zeros_like(forward_data), forward_data) + 1e-8)
                    loss = 0.5 * (loss_X + loss_data)
                    acc = torch.tensor([0.0])
                    
            total_loss += loss.item() * actual_bs
            total_acc += acc.item() * actual_bs
            
        # Return averages[cite: 1]
        return total_loss / total_batches, total_acc / total_batches

    # 1. Collect Data
    results_data = []
    
    # Ensure there are exactly 5 operators
    num_ops = min(len(train_ops), 5) 
    
    for i in range(num_ops):
        train_op = train_ops[i]
        test_op = test_ops[i]
        
        # Name formulation
        op_label = train_op[1] if isinstance(train_op, list) else f"Operator {i+1}"
        
        # Evaluate dynamically[cite: 1]
        train_loss, train_acc = evaluate(train_loader, train_op, train_loader.dataset)
        test_loss, test_acc = evaluate(test_loader, test_op, test_loader.dataset)
        
        # Format strings for the table
        if args.classify:
            results_data.append([op_label, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{test_loss:.4f}", f"{test_acc:.4f}"])
        else:
            results_data.append([op_label, f"{train_loss:.4f}", "N/A", f"{test_loss:.4f}", "N/A"])

    # 2. Plot the Table using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off') # Hide axes
    
    columns = ["Operator", "Train Loss", "Train Acc", "Test Loss", "Test Acc"]
    
    # Define Column Colors: Train columns in light blue, Test columns in light green
    header_colors = ["#f2f2f2", "#cce5ff", "#cce5ff", "#d4edda", "#d4edda"]
    cell_colors = [["#ffffff", "#e6f2ff", "#e6f2ff", "#e8f5e9", "#e8f5e9"] for _ in range(num_ops)]

    table = ax.table(
        cellText=results_data,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=header_colors,
        cellLoc='center',
        loc='center'
    )
    
    # Formatting table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8) # Adjust column width and row height
    
    # 3. Save the Table as JPEG
    os.makedirs('models', exist_ok=True) # Ensure models directory exists[cite: 1]
    save_name = getattr(args, 'project_name', 'operator_comparison')
    save_path = os.path.join('models', f"{save_name}.jpeg")
    
    plt.savefig(save_path, format='jpeg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Comparison table successfully saved to: {save_path}")