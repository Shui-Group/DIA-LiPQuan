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
This web provides a general pipeline for analyzing DIA LiP data using the DIA-LiPQuan package.
usage page:http://10.19.26.62:9600/usage.html
The example files are located in the "path_to/DIA-LiPQuan/example_data" directory.

