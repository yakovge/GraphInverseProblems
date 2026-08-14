import subprocess

dic = {
    'epochs': 1000,
    'head_epochs': 40,
    'test_head_epochs': 10,
    'noise': True,
    'painting': True,
    'blurring': True,
    'sensoring': True,
    'pdessm': False,
    'method': 'foundation',
    'classify': 0,
    'project_name': 'epoch_1000_head_40_test_10_pdessm_test'
}

def create_message(dic):
    message = 'python src/main_3_linear_inv_problems.py'
    for key, value in dic.items():
        message += f' --{key} {value}'
    return message

def run_commands_sequentially(commands):
    """
    Runs a list of terminal commands one after the other.
    Waits for each command to finish before moving to the next.
    """
    for index, command in enumerate(commands, start=1):
        print(f"\n--- [Step {index}] Running: {command} ---")
        
        # shell=True allows running exact terminal strings
        # capture_output=True grabs the stdout and stderr
        # text=True returns the output as a string instead of bytes
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Print the output from the terminal
        if result.stdout:
            print(f"Output:\n{result.stdout.strip()}")
            
        # Print any errors if they occurred
        if result.stderr:
            print(f"Errors:\n{result.stderr.strip()}")
            
        print(f"--- Finished with exit code: {result.returncode} ---")

# --- Example Usage ---
if __name__ == "__main__":
    # Define the lines you want to run
    my_commands = [
        create_message(dic)
    ]
    dic['pdessm'] = True
    dic['sensoring'] = False
    dic['project_name'] = 'epoch_1000_head_40_test_10_sensoring_test'
    my_commands.append(create_message(dic))
    dic['sensoring'] = True
    dic['blurring'] = False
    dic['project_name'] = 'epoch_1000_head_40_test_10_blurring_test'
    my_commands.append(create_message(dic))
    dic['blurring'] = True
    dic['painting'] = False
    dic['project_name'] = 'epoch_1000_head_40_test_10_painting_test'
    my_commands.append(create_message(dic))
    dic['painting'] = True
    dic['noise'] = False
    dic['project_name'] = 'epoch_1000_head_40_test_10_noise_test'
    my_commands.append(create_message(dic))
    
    run_commands_sequentially(my_commands)

