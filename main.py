import subprocess
import sys

def main():
    print("=== Training Random Forest (Tabular Data) ===")
    subprocess.run([sys.executable, "train_rf.py"], check=True)
    
    print("\n=== Training CNN (Image Data) ===")
    subprocess.run([sys.executable, "train_cnn.py"], check=True)

if __name__ == "__main__":
    main()
