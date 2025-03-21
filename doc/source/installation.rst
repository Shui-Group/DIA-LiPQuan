Installation
===========

Install lipana from the repository:

.. code-block:: bash

   pip install git+https://github.com/Shui-Group/LiPAna.git

Or clone the repository and include it to system path:

.. code-block:: python

   import sys
   sys.path.append("/path/to/LiPAna")
   import lipana

.. Or with pip:

.. .. code-block:: bash

..    pip install lipana


To use maxLFQ and limma, R and associated packages should be installed.

.. code-block:: R

   install.packages("arrow")
   install.packages("iq")
   if (!require("BiocManager", quietly = TRUE))
      install.packages("BiocManager")
   BiocManager::install("limma")
