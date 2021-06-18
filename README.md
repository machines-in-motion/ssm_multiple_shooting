# Learning Dynamical Systems from Noisy Sensor Measurements using Multiple Shooting

A Python implementation of the multiple shooting framework and all the code necessary to reproduce the experiments of the paper are provided. The core of the multiple shooting implementation is in the file named ssm.py.

### Create environment
    $ cd ssm_ms/
    $ conda env create -f environment.yml
    $ conda activate learning

### Creata dataset
    $ cd data/ 
    $ mkdir datasets/
    $ python n_link.py --n_pendulum 1
    $ python n_link.py --n_pendulum 4
    $ bash moving_mnist.sh
    $ python moving_mnist_data.py
    $ python lorenz_data.py
    $ cd ..

### Simple and quadruple pendulum
    $ python pendulum.py --ll 0
    $ python pendulum.py --ll 1
    $ python quad_link.py --ll 0
    $ python quad_link.py --ll 1

### Moving MNIST 
    $ python moving_mnist.py
    $ python moving_mnist_test.py

### Lorenz

Train SSM:

    $ python lorenz.py --T 10 
    $ python lorenz.py --T 100
    $ python lorenz.py --T 1000
    $ python lorenz.py --T 10000

Evaluation with a UKF:

    $ python lorenz_test.py --T 10
    $ python lorenz_test.py --T 100
    $ python lorenz_test.py --T 1000
    $ python lorenz_test.py --T 10000

Evaluation with a RNN:
    
    $ python lorenz_test_rnn.py --T 10
    $ python lorenz_test_rnn.py --T 100
    $ python lorenz_test_rnn.py --T 1000
    $ python lorenz_test_rnn.py --T 10000


