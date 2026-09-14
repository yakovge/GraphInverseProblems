import subprocess

base_dic = {
    'epochs': 1000,
    'head_epochs': 40,
    'test_head_epochs': 10,
    'noise': True,
    'painting': True,
    'blurring': True,
    'sensoring': True,
    'pdessm': True,
    'method': 'foundation',
    'cglsIter': 5,
    'solveIter': 5,
    'classify': 0,
    'max_patience': 150,
}

# Five leave-one-out experiments
held_out_ops = ['noise', 'painting', 'blurring', 'sensoring', 'pdessm']

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
    commands = []

    for held_out in held_out_ops:
        # Control (denoising_bypass=False)
        dic = base_dic.copy()
        dic['held_out_op'] = held_out
        dic['denoising_bypass'] = False
        dic['project_name'] = f'control_holdout_{held_out}'
        commands.append(create_message(dic))

        # Treatment (denoising_bypass=True)
        dic = base_dic.copy()
        dic['held_out_op'] = held_out
        dic['denoising_bypass'] = True
        dic['project_name'] = f'treatment_holdout_{held_out}'
        commands.append(create_message(dic))

    run_commands_sequentially(commands)
