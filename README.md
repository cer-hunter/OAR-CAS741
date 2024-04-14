# OAR - Optical Alphabet Recognition

Developer Names: Hunter Ceranic

Date of project start: January 15, 2023

This project is an Optical Character Recognition program for Capital letters, using logistic regression models, designed from scratch.
While not the most efficient model possible this is intended to be an educational project to learn about the theory behind Logistic Regression and machine learning and how this theory is implemented in practice.

The folders and files for this project are as follows:

docs - Documentation for the project
refs - Reference material used for the project, including papers
src - Source code
tests - Test cases
.circleci - CircleCI folder for continous integration configuration

The recommended order to read the documentation is
```mermaid
graph LR
    E(SRS) --> D
    D(VnV Plan) --> C
    C(MG) --> B
    B(MIS) --> A(VnV Report)
```

---

## Install

See [INSTALL.md](./INSTALL.md) for details on how to install and get started with this project on your own computer.

## Usage

To run the OAR classifier program, in Anaconda Powershell 'cd' into the 'src/' folder in the repo and run the command 'python main.py'.
To run the model training, adjust the training specs of 'oarTest.py' to your desire in a text editor of your choice (VScode is recommended), and in Anaconda Powershell 'cd' into the 'src/' folder in the repo, and run the command 'python oarTest.py'.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to contribute to this project.

## Troubleshooting and Comments

This section will be updated to inform potential users of known issues and common solutions.


