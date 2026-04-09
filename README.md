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
import sys
sys.path.append("/path/to/LiPAna")
import lipana

###R Dependencies (for maxLFQ and limma)

To use maxLFQ and limma, you need R and the following packages installed:

install.packages("arrow")
install.packages("iq")

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("limma")
