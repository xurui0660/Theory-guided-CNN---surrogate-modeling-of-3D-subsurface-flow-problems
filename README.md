# Theory-guided-CNN---surrogate-modeling-of-3D-subsurface-flow-problems

We develop a theory-guided CNN (encoder-decoder) for surrogate modeling of 3D subsurface transient hydrological problems, particularly single phase flow with consideration of vertical wells that penetrate the formation at arbitray lengths. The codes are written in Python and applied to Pytorch platform, and are designed to run in parallel on GPUs. The network takes the image-like 3D permeability field and a 3D time matrix as inputs, and outputs the corresponding pressure field as 3D images. An accurate surrogate model can be constructed using limited training dataset with proper incorporation of theoretial guidance (by defining a proper loss function composing the residual of the governing equation, initial and boundary conditions, as well as the data mismatch). Once properly trained, the surrogate model can be used for efficient solution of common hydrological problems such as uncertainty quantification, inverse modeling, or optimization tasks, which are highly computationally-demanding using conventional numerical simulation tools.

An example is given, which has four producing wells located at the four corners of the formation producing at constant bottom hole pressure. The permeability fields are generated using KL expansion assuming known mean, variance, and correlation length data. The trained surrogate model can provide accurate and fast estimation of the pressure field for a random permeability field at any given time. The well production rate can then be calculated based on the predicted pressure using Peaceman's formula. Good agreement between the surrogate model and the simulation tool (UNCONG) is observed. Uncertainty quantification and inverse modeling tasks are performed. Compared with the numerical simulation tool (UNCONG), the surrogate model demonstrates comparable accuracy but significantly improved computational efficiency (up to hundreads of times faster). Details of this example can be found in our work （https://doi.org/10.48550/arXiv.2111.08691) and related data can be provided upon request (contact Rui Xu at xur@pcl.ac.cn).

To run the code, type the following onto the command line (assuming linux machine with at least one GPU): 
python -m torch.distributed.launch --nproc_per_node=1 /YOUR_CODE_PATH/TgCNN_train_1phase_10_220_60_constp.py

Feel free to use and modify the codes. Please provide citation of the codes and the paper above if you use them in your work. Feel free to contact the authors in case you have trouble or comments about the codes.
