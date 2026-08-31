run_demo.py
import subprocess
import sys
import os

def get_resource_path(path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.path.abspath("."), path)

def main():
    seq = [train.py,
    evaluate_direct.py,
    train_context.py,
    evaluate_context.py,
    shuffle_check.py]
    for script in seq:
        script_path = get_resource_path(script)
        print("running: {script}")
        result = subprocess.run([sys.executable, script_path], check = True)

        if result.returncode != 0:
            print("44 notfoundindasystem 44 da new eraera (na srsly ya failed. Halting execution bish)")
            sys.exit(result.returncode)

if __name__ == "__main__":
    main()