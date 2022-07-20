# CatchBackdoor: Backdoor Testing via Critical Trojan Neural Path Fuzzing

# Setup
The code should be run using python 3.6.0, Tensorflow-gpu 2.4.0, Keras 2.4.3, Foolbox 2.3.0, PIL, h5py, and adversarial-robustness-toolbox

# Datasets & Models

Here, we give LeNet-5 trained under MNIST as a example.

# How to train a trojaned model
 - BadNets model
  ```python
 python trian_model.py
  ```
  
# How to run
- With benign examples
 ```python
 python CatchBackdoor.py
  ```
  
- Without benign examples
 ```python
python CatchBackdoor-noise.py
  ```

- visualization
 ```python
python visualization.py
  ```

# Note
See the paper CatchBackdoor: Backdoor Testing via Critical Trojan Neural Path Fuzzing for more details.
