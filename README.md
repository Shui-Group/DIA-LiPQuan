# DIA-LiPQuan

An informatics pipeline tailored to DIA LiP-MS quantification and downstream analysis.

---

## Installation

### Install from GitHub

You can install LiPAna directly via `pip`:

```bash
pip install git+https://github.com/Shui-Group/DIA-LiPQuan.git
```
Or clone the repository and add it to your Python path
```
git clone https://github.com/Shui-Group/DIA-LiPQuan.git
```
```
import sys
sys.path.append("/path/to/DIA-LiPQuan")
import lipana
```
R Dependencies (for maxLFQ and limma)

To use maxLFQ and limma, you need R and the following packages installed:
```
install.packages("arrow")
install.packages("iq")
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("limma")
```
---
## Usages

### overview
This page provides a general pipeline for analyzing Lipidomics (LiP) data using the lipana package.
The example data used here is a truncated search report from DIA-NN. This dataset includes three experiment conditions, each with three replicates, and only 1000 proteins are retained.
The example files are located in the "path_to/DIA-LiPQuan/example_data" directory.

```
import gzip
import shutil
from pathlib import Path

workspace = Path(".").resolve().parents[1].joinpath("example_data")
print("current workspace:", str(workspace))
# Unzip gzipped files
for file in workspace.glob("*.gz"):
    with gzip.open(file, "rb") as f_in:
        with open(file.with_suffix(""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
```

