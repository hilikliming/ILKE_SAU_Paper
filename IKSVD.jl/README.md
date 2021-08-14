# IKSVD.jl
This is a Julia Implementation of the Incremental K-SVD Dictionary Learning Presented in
"IK-SVD: Dictionary Learning for Spatial Big Data via Incremental Atom Update" by Lizhe Wang ; Ke Lu ; Peng Liu ; Rajiv Ranjan and Lajiao Chen: 
[paper](https://ieeexplore.ieee.org/document/6799952)

This implementation is based on [Ishita Takeshi's Julia implementation of K-SVD](https://github.com/IshitaTakeshi/KSVD.jl)

This implementation assumes that you have already trained on Y_1 ... Y_{s-1} and you wish to solve m more atoms using dataset Y_s, as a result, atoms d_1...d_n in D_old will be unchanged and updates will only be made to the d_{n+1}...d_{n+m} atoms according to step 3 of the IK-SVD (note k \in \{n+1, ..., n+m\} )


Conditioned USPS dataset can be found [here.](https://www.dropbox.com/sh/4pn9kty1kw8dl67/AADa7YshuXCtW5ouYxcc-Rnpa?dl=0)

Or just take the normal USPS data which is 16x16 grayscale... invert the colors, resample to 20x20 and put 4 columns of zero padding on either side and 4 rows above and below the image resulting in images that are 28x28 with 4 pixels of zero padding on all sides just like the MNIST digits.
