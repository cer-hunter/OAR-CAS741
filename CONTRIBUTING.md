# Contributing

Welcome. Bug fixes, comments, and questions are appreciated.

## Code of Conduct

The project is governed by the [Code of Conduct](CodeOfConduct.md). All participants are expected to uphold this code.

---

## Useful Information

See [INSTALL](INSTALL.md) for getting setup.

---

## How to contribute

### Submitting a bug report

All bug reports are to be created in [Issues](https://github.com/cer-hunter/OAR-CAS741/issues). Explain the problem and include additional details to help maintainers reproduce the problem:

- Use a clear and descriptive title for the issue to identify the problem.
- Describe the exact steps which reproduce the problem in as many details as possible. When listing steps, don't just say what you did, but explain how you did it.
- Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
- Explain which behavior you expected to see instead and why.
- Include details about your environment.

### Submitting changes

A good pull request speeds up the development process, please:

- Create a pull request that details the issue you are solving and why. Provide a link to the issue, and other relavent information.
- Add testing results, with linked files or screenshots.
- Ensure you follow the style guidelines found below.

---

## Styleguides

### Git commit messages

### Python Code

This repo uses the Python [Flake8 Rules](https://www.flake8rules.com/). In addition --extend-ignore E402 is used when running the linter.

The Flake8 linter is required, for VSCode users:

    Name: Flake8
    Id: ms-python.flake8
    Description: Linting support for Python files using Flake8.
    Version: 2023.13.10681010
    Publisher: Microsoft
    VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-python.flake8
    