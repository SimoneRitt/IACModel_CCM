# Computational Cognitive Modeling 
## Fall 2024

### Interactive Activation and Competition (IAC) Network

This project seeks to implement an interactive visualization of James McClelland's IAC network. This network represents human memory and can be used to illustrate several interesting properties of human cognition.

#### Getting Started

For convenience, environment requirements are specified in `requirements.txt`. To set up an IAC conda environment, follow these steps:
- `conda create -n IAC python=3.10.15`
- `conda activate IAC`
- Navigate to the directory `IACModel_CCM`
- `pip install -r requirements.txt`

To create a Jupyter kernel called `IAC`, run: `python -m ipykernel install --name IAC --user`

To uninstall the Jupyter kernel, run: `jupyter kernelspec uninstall IAC` 
- Note that the kernel name may be slightly different. You can check this with `jupyter kernelspec list`
