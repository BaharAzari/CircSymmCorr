[![arXiv](https://img.shields.io/badge/arXiv-2009.05135-b31b1b.svg)](https://arxiv.org/abs/2107.12480)
# Circular-Symmetric Correlation Layer based on FFT

Despite the vast success of standard planar convolutional neural networks, they are not the most efficient choice for analyzing signals that lie on an arbitrarily curved manifold, such as a cylinder. The problem arises when one performs a planar projection of these signals and inevitably causes them to be distorted or broken where there is valuable information. We propose a Circular-symmetric Correlation Layer (CCL) based on the formalism of roto-translation equivariant correlation on the continuous group S1×ℝ, and implement it efficiently using the well-known Fast Fourier Transform (FFT) algorithm. We showcase the performance analysis of a general network equipped with CCL on various recognition and classification tasks and datasets. 

