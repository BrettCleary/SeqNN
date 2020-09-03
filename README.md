# SeqNN
A simple sequential neural network python extension

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project
SeqNN is a simple single threaded sequential neural network python extension written in c++ and compiled with SWIG.

### Features
- Modular layers
    - 2 Dimensional Convolutional
    - 2 Dimensional Max/Min Pool
    - Fully Connected Dense Layer
- Regularization
    - Weight decay
    - Soft weight sharing for the 2D Convolutional layer
    - Early Stopping

### Built With
* [SWIG](http://www.swig.org/download.html)

<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
```sh
git clone https://github.com/BrettCleary/DigitCNN
```
OR
2. Install with PyPi
```sh
pip install SeqNN
```

<!-- USAGE EXAMPLES -->
## Usage

Example usage with a subset of the MNIST dataset is given in DigitCNN.py. Here is a sample output:

    Error Rate (%) after training  2  number of epochs is  22.8
    Error Rate (%) after training  4  number of epochs is  14.4
    Error Rate (%) after training  6  number of epochs is  12.8
    Error Rate (%) after training  8  number of epochs is  12.0
    Error Rate (%) after training  10  number of epochs is  10.8
    Error Rate (%) after training  12  number of epochs is  10.8
    Error Rate (%) after training  14  number of epochs is  10.0
    Error Rate (%) after training  16  number of epochs is  10.8
    Error Rate (%) after training  18  number of epochs is  10.4
    Error Rate (%) after training  20  number of epochs is  9.6

    The error rate for test dataset is  6.4


<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Project Link: [https://github.com/BrettCleary/SeqNN](https://github.com/BrettCleary/SeqNN)
