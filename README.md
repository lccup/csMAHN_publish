# csMAHN

# Create conda environment

```shell
hostnamectl
#   ...
#   Operating System: CentOS Linux 7 (Core)
#        CPE OS Name: cpe:/o:centos:centos:7
#             Kernel: Linux 3.10.0-957.el7.x86_64
#       Architecture: x86-64
#   ...

conda --version
# conda 24.1.2

conda config --add channels bioconda
conda config --add channels conda-forge
conda config --show channels
# channels:
#   - conda-forge
#   - bioconda
#   - defaults
#

# create the environment
conda activate base
conda create --name publish --yes
conda activate publish
conda install --name publish --file require_publish --yes
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

pip install samap
pip install came
conda list --name publish | grep -E "^(python )|(scanpy)|(torch)|(r-base )|(r-seurat)|(r-seuratobject)|(came)|(^samap)"
# python                    3.9.18          h0755675_1_cpython    conda-forge
# pytorch                   2.1.2           cpu_generic_py39h163b580_0    conda-forge
# scanpy                    1.9.3              pyhd8ed1ab_0    conda-forge
# samap                     1.0.15                   pypi_0    pypi
# came                      0.1.13                   pypi_0    pypi
# r-base                    4.3.3                hb8ee39d_0    conda-forge
# r-seurat                  5.0.0             r43ha503ecb_0    conda-forge
# r-seuratobject            5.0.1             r43ha503ecb_0    conda-forge




# create jupyter kernel
conda activate publish
python -m ipykernel install --user --name 'publish' --display-name 'publish'
##  [r-kernel](https://irkernel.github.io/installation/#linux-panel)
echo 'IRkernel::installspec(user = TRUE,name="R_publish",display="R_publish")' > temp_r_script.r
Rscript temp_r_script.r
rm temp_r_script.r
## check jupyter kernel
conda activate
jupyter kernelspec list
# Available kernels:
#   ...
#   publish           ....local/share/jupyter/kernels/publish
#   r_publish         ....local/share/jupyter/kernels/r_publish
#   ...

```

We provide a small dataset and tow scripts, `demo.ipynb` and `demo_Seurat.ipynb` to test if the environment is valid.
