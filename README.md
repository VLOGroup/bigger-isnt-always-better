# Bigger Isn't Always Better
![Concept](/concept/concept.jpg)

Official Github Repository of "Bigger Isn’t Always Better: Towards a General Prior for Medical Image Reconstruction" from GCPR 2024.
> **Abstract:**
>
> Diffusion model have been successfully applied to many inverse problems, including MRI and CT reconstruction. Researchers typically re-purpose models originally designed for unconditional sampling without modifications. Using two different posterior sampling algorithms, we show empirically that such large networks are not necessary. Our smallest model, effectively a ResNet, performs almost as good as an attention U-Net on in-distribution reconstruction, while being significantly more robust towards distribution shifts. Furthermore, we introduce models trained on natural images and demonstrate that they can be used in both MRI and CT reconstruction, out-performing model trained on medical images in out-of-distribution cases. As a result of our findings, we strongly caution against simply re-using very large networks and encourage researchers to adapt the model complexity to the respective task. Moreover, we argue that a key step towards a general diffusion-based prior is training on natural images.

Modified from Chung & Ye [1] ([Code](https://github.com/HJ-harry/score-MRI)) and Jalal et al. [2] ([Code](https://github.com/utcsilab/csgm-mri-langevin)).

## Training
Training scripts for MRI, CT and natural images are provided in ```training_scripts```.

## Evaluation
Scripts in ```evaluation``` demonstrate how the reconstructions are calculated. ```hist.sh``` creates the image statistics shown in Fig. 1.

## Cite This Work
Please consider citing this work using
```
@inproceedings{Glaszner2024,
	title={Bigger Isn't Always Better: Towards a General Prior for MRI Reconstruction}, 
	author={Lukas Glaszner and Martin Zach},
	year={2024},
	booktitle={Pattern Recognition, DAGM GCPR 2024}
}
```


## References
[1] Chung, H., Ye, J.C.: Score-based diffusion models for accelerated MRI. Medical Image Analysis 80, 102479 (2022).

[2] Jalal, A., et al.: Robust compressed sensing MRI with deep generative priors. In: Advances in Neural Information Processing Systems. vol. 34, pp. 14938–14954. Curran Associates, Inc. (2021).

[3] Song, Y., et al.: Score-based generative modeling through stochastic differential equations. In: International Conference on Learning Representations (2021).
