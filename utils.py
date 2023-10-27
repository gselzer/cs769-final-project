import os
import subprocess
import platform

class WordAligner:
    def __init__(self):
        system_name = platform.system()
        self.eflomal_dir = "maceflomal/" if system_name == "Darwin" else "eflomal/"
        
        if not os.path.exists(self.eflomal_dir):
            if system_name == "Darwin":
                self.__setup_macos()
            else:
                self.__setup_other()

    def __setup_macos(self):
        # Install Homebrew if not already installed
        if subprocess.run(["which", "brew"]).returncode != 0:
            subprocess.run(["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"], check=True)

        # Install LLVM if not already installed
        if subprocess.run(["brew", "list", "llvm"]).returncode != 0:
            subprocess.run(["brew", "install", "llvm"], check=True)

        subprocess.run(["git", "clone", "https://github.com/huseinzol05/maceflomal"], check=True)
        self.__fix_macos_makefile()
        subprocess.run("cd maceflomal && export CC=/usr/local/bin/gcc-11 && make", shell=True, check=True)
        subprocess.run(["sudo", "make", "install"], cwd="maceflomal", check=True)
        subprocess.run(["python3", "setup.py", "install"], cwd="maceflomal", check=True)

    def __setup_other(self):
        subprocess.run(["git", "clone", "https://github.com/robertostling/eflomal"], check=True)
        subprocess.run(["make"], cwd="eflomal", check=True)
        subprocess.run(["sudo", "make", "install"], cwd="eflomal", check=True)
        subprocess.run(["python3", "setup.py", "install"], cwd="eflomal", check=True)

    def __fix_macos_makefile(self):
        # Define the directory of the Makefile
        if platform.system() == "Darwin":
            makefile_dir = os.path.join(os.getcwd(), "maceflomal")
        else:
            makefile_dir = os.path.join(os.getcwd(), "eflomal")

        makefile_path = os.path.join(makefile_dir, "Makefile")

        # Read the Makefile
        with open(makefile_path, "r") as f:
            content = f.read()

        # Replace spaces with tabs
        content = content.replace("    ", "\t")

        # Make other replacements
        content = content.replace("CFLAGS=-Ofast -march=native -Wall --std=gnu99 -Wno-unused-function -g -fopenmp",
                                  "CC=/opt/homebrew/Cellar/llvm/17.0.3/bin/clang\nCFLAGS=-Ofast -Wall --std=gnu99 -Wno-unused-function -g -fopenmp")
        content = content.replace("LDFLAGS=-lm -lgomp -fopenmp",
                                  "LDFLAGS=-L/opt/homebrew/Cellar/llvm/17.0.3/lib -Wl,-rpath,/opt/homebrew/Cellar/llvm/17.0.3/lib -lm -lomp -fopenmp")

        # Write the modified content back to the Makefile
        with open(makefile_path, "w") as f:
            f.write(content)

    def word_alignments(self, source_file, target_file, output_file):
        subprocess.run([
            "python", os.path.join(self.eflomal_dir, "align.py"), 
            "-s", source_file, 
            "-t", target_file, 
            "-f", output_file, 
            "--model", "1", 
            "--source-prefix", "0", 
            "--source-suffix", "0",
            "--target-prefix", "0",
            "--target-suffix", "0",
            "--overwrite"
        ], check=True)

