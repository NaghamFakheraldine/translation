import subprocess

def run_script(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"{script_name} output:\n{result.stdout}")

if __name__ == "__main__":
    run_script("../Scripts/get_data.py")
    run_script("../Scripts/tokenizer.py")
    run_script("../Scripts/train_model.py")
