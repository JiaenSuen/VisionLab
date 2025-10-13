import sys
import subprocess

def main():
    print("Option : ")
    print("1. train")
    print("2. test")
    choice = input("Select : ").strip().lower()

    if choice == "train":
        module_name = "TrainingSection.train"
    elif choice == "test":
        module_name = "TrainingSection.test"
    else:
        print(f"unknow: {choice}")
        sys.exit(1)

    subprocess.run([sys.executable, "-m", module_name])

if __name__ == "__main__":
    main()
