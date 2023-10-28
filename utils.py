import os
import subprocess
import platform
import eflomal

class WordAligner:
    def __init__(self):
        self.system_name = platform.system()
        self.eflomal_installed = False

        if not self.eflomal_installed:
            if self.system_name == "Darwin":
                self.__setup_macos()
                self.eflomal_installed = True
            else:
                self.__setup_other()
                self.eflomal_installed = True

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

    def __setup_other(self):
        subprocess.run(["pip", "install", "eflomal==1.0.0b0"], capture_output=True, text=True)

    def word_alignments(self, source_file, target_file, output_file):
        if self.system_name == "Darwin":
            subprocess.run([
                "python", os.path.join("maceflomal/", "align.py"),
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
        else:
            aligner = eflomal.Aligner(model = 1)

            with open(source_file, 'r', encoding='utf-8') as src_data, \
                 open(target_file, 'r', encoding='utf-8') as trg_data:
                aligner.align(src_input = src_data,
                              trg_input = trg_data,
                              links_filename_fwd = 'links.fwd',
                              links_filename_rev = 'links.rev')

            with open(source_file, 'r', encoding='utf-8') as src_data, \
                 open(target_file, 'r', encoding='utf-8') as trg_data, \
                 open('links.fwd', 'r', encoding='utf-8') as fwd_links, \
                 open('links.rev', 'r', encoding='utf-8') as rev_links, \
                 open('priors', 'w', encoding='utf-8') as priors_f:
                priors_tuple = eflomal.calculate_priors(src_data, trg_data, fwd_links, rev_links)
                eflomal.write_priors(priors_f, *priors_tuple)

    def joint_dropout(self):
        # confused: phrase verse word verse token
        # do not allow two adjacent phrases
        # all phrases, regardless of length, are potential candidates for substitution
        # a maximum of 10 variables are allowed in each sentence
        # apply BPE segmentation after joint dropout. Make sure variables are not split into smaller segments
        # Joint Dropout Rate: (optimal JDR=0.3) 
        pass

