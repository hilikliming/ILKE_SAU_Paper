# ILKE_SAU_Paper
Repo for Code from: Adaptive Classification Using Incremental Linearized Kernel Embedding

Main test script is titled: testEasyLKE-MNIST.jl, Read comments to see how to set flags to learn and save a new model to .jld format.

test_new_SRC_save_load_D.jl gives a simple demonstration of saving a model and reloading it from it's .jld

Arrowhead.jl is a slightly modified version of the Julia module available on Ivan Slapinčar's github which features the algorithms from his joint work with NJ Stor and JL Barlow in "Accurate eigenvalue decomposition of real symmetric arrowhead matrices and applications" (what a great paper!!!).


/backup which contains the USPS data properly formatted can be found at this link (too large to host on GitHub): https://www.dropbox.com/sh/arpwnrl6t1mm6qe/AAA3Dmus9tUObwHR9LD6j6Zta?dl=0

after downloading the contents of directory 'backup' (USPS data and labels) add the directory to the ILKE_SAU_Paper repo folder and run the main test script.
