# Computational Cognitive Modeling 
## Fall 2024

### Interactive Activation and Competition (IAC) Network

This project seeks to implement an interactive visualization of James McClelland's IAC network. This network represents human memory and can be used to illustrate several interesting properties of human cognition. 

Directory:
- `data/` contains McClelland's original West Side Story inspired Jets vs. Sharks dataset (`jets_sharks.csv`), as well as several novel application datasets.
- `IAC_plotting.py` contains the core visualization functions and can be imported into Jupyter Notebooks for interactive use.
  - Within this script, `plot(df, hidden_state=None)` takes in a Pandas DataFrame and a string column name and renders the visualization
- `IAC_Example.ipynb` contains example implementations of the network.
