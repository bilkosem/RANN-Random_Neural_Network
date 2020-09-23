# Random Neural Network (RANN) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

**Queueing Recurrent Neural Network (Q-RNN)** is a new kind of Artificial Neural Network that has been designed to use in time-series forecasting applications. According to experiments that have been run, QRNN has a potential to outperform the LSTM, Simple RNN and GRU, in the cases where the dataset has highly non-linear characteristics.

# Table of contents

- [What is RANN?](#RANN)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## What is RANN? 🤔
![Random Neuron](images/random_neuron.png)

## Installation 🛠

Installing via [pip](https://pip.pypa.io/en/stable/) package manager:

```bash
pip install rann
```

Installing via GitHub:

```bash
git clone https://github.com/bilkosem/random_neural_network
cd rann
python setup.py install
```

## Usage 👩‍💻

```python
from rann import RANN

data=data.reshape((samples,features))
rann = RANN([features, hidden neurons, output neurons]) # Shape of the network

for s in range(samples):
    rann.feedforward()
    # Calculate Loss
    rann.backpropagation()
```
Check the [examples](https://github.com/bilkosem/rann/tree/master/examples) folder to see detailed use 🔎.
## References 📚

[1] [Gelenbe, Erol. (1989). Random Neural Networks with Negative and Positive Signals and Product Form Solution. Neural Computation - NECO. 1. 502-510. 10.1162/neco.1989.1.4.502.](https://www.researchgate.net/publication/239294946_Random_Neural_Networks_with_Negative_and_Positive_Signals_and_Product_Form_Solution) 

[2] [Gelenbe, Erol. (1993). Learning in the Recurrent Random Neural Network. Neural Computation. 5. 154-164. 10.1162/neco.1993.5.1.154.](https://www.researchgate.net/publication/220499635_Learning_in_the_Recurrent_Random_Neural_Network)

[3] [Basterrech, S., & Rubino, G. (2015). Random Neural Network Model for Supervised Learning Problems. Neural Network World, 25, 457-499.](https://www.semanticscholar.org/paper/Random-neural-network-model-for-supervised-learning-Basterrech-Rubino/b2ebb88e1d78c726aab274ec149d65e86999cbef)

[4] [Hossam Abdelbaki (2020). rnnsimv2.zip (https://www.mathworks.com/matlabcentral/fileexchange/91-rnnsimv2-zip), MATLAB Central File Exchange. Retrieved September 22, 2020.](https://www.mathworks.com/matlabcentral/fileexchange/91-rnnsimv2-zip?s_tid=FX_rc1_behav)


## License

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
