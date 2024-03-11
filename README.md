<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Spiking-FullSubNet</h3>

  <p align="center">
    Intel N-DNS Challenge Algorithmic Track Winner
    <br />
    <a href="https://haoxiangsnr.github.io/spiking-fullsubnet/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/haoxiangsnr/spiking-fullsubnet/">View Demo</a>
    ·
    <a href="https://github.com/haoxiangsnr/spiking-fullsubnet/issues">Report Bug</a>
    ·
    <a href="https://github.com/haoxiangsnr/spiking-fullsubnet/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

![Spiking-FullSubNet](./docs/source/images/project_image.png)

We are proud to announce that Spiking-FullSubNet has emerged as the winner of Intel N-DNS Challenge Track 1 (Algorithmic). Please refer to our [brief write-up here](./Spiking-FullSubNet.pdf) for more details. This repository serves as the official home of the Spiking-FullSubNet implementation. Here, you will find:

- A PyTorch-based implementation of the Spiking-FullSubNet model.
- Scripts for training the model and evaluating its performance.
- The pre-trained models in the `model_zoo` directory, ready to be further fine-tuned on the other datasets.

<!---
We are actively working on improving the documentation, fixing bugs and removing redundancies. Please feel free to raise an issue or submit a pull request if you have suggestions for enhancements.
Our team is diligently working on a comprehensive paper that will delve into the intricate details of Spiking-FullSuNet's architecture, its operational excellence, and the broad spectrum of its potential applications. Please stay tuned!
-->

## Updates

[2024-02-26] Currently, our repo contains two versions of the code:

1. The **frozen version**, which serves as a backup for the code used in a previous competition. However, due to a restructuring in the `audiozen` directory, this version can no longer be directly used for inference. If you need to verify the experimental results from that time, please refer to this specific commit: [38fe020](https://github.com/haoxiangsnr/spiking-fullsubnet/tree/38fe020cdb803d2fdc76a0df4b06311879c8e370). There you will find everything you need. After switching to this commit, you can place the checkpoints from the `model_zoo` into the `exp` directory and use `-M test` for inference or `-M train` to retrain the model.

2. The **latest version** of the code has undergone some restructuring and optimization to make it more understandable for readers. We've also introduced `acceleate` to assist with better training practices. We believe you can follow the instructions in the help documentation to run the training code directly. The pre-trained model checkpoints and a more detailed paper will be released by next weekend, so please stay tuned for that.



## Documentation

See the [Documentation](https://haoxiangsnr.github.io/spiking-fullsubnet/) for installation and usage. Our team is actively working to improve the documentation. Please feel free to raise an issue or submit a pull request if you have suggestions for enhancements.

## License

All the code in this repository is released under the [MIT License](https://opensource.org/licenses/MIT), for more details see the [LICENSE](LICENSE) file.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/haoxiangsnr/spiking-fullsubnet.svg?style=for-the-badge
[contributors-url]: https://github.com/haoxiangsnr/spiking-fullsubnet/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/haoxiangsnr/spiking-fullsubnet.svg?style=for-the-badge
[forks-url]: https://github.com/haoxiangsnr/spiking-fullsubnet/network/members
[stars-shield]: https://img.shields.io/github/stars/haoxiangsnr/spiking-fullsubnet.svg?style=for-the-badge
[stars-url]: https://github.com/haoxiangsnr/spiking-fullsubnet/stargazers
[issues-shield]: https://img.shields.io/github/issues/haoxiangsnr/spiking-fullsubnet.svg?style=for-the-badge
[issues-url]: https://github.com/haoxiangsnr/spiking-fullsubnet/issues
[license-shield]: https://img.shields.io/github/license/haoxiangsnr/spiking-fullsubnet.svg?style=for-the-badge
[license-url]: https://github.com/haoxiangsnr/spiking-fullsubnet/blob/master/LICENSE.txt
