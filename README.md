# Bigger Isn't Always Better
![Concept](/concept/concept.jpg)

Official Github Repository of "[Bigger Isn’t Always Better: Towards a General Prior for Medical Image Reconstruction](https://doi.org/10.1007/s11263-026-02904-1)" from the [International Journal of Computer Vision (IJCV, 2026)](https://doi.org/10.1007/s11263-026-02904-1) and [German Conference on Pattern Recognition (GCPR 2024)](https://doi.org/10.1007/978-3-031-85181-0_18).
> **Abstract:**
>
> Diffusion models play an important role in many state-of-the-art algorithms to solve inverse problems in imaging. In particular, reconstructing MRI and CT images from undersampled data can be reformulated as a canonical inverse problem that benefits from strong priors. Researchers typically re-purpose models originally designed for unconditional sampling without modifications and achieve remarkable accuracy on in-distribution (ID) reconstruction. However, due to the scarce availability of training data and the large number of different imaging setups, increasing the generalization capabilities of diffusion-based priors is key to clinical adoption. To do so, we propose two solutions: (i) using smaller models and (ii) training on natural images. Using three different posterior sampling algorithms, we evaluate the influence of network size and training data. Our smallest model, effectively a ResNet, performs almost as good as an attention U-Net on ID reconstruction, while being significantly more robust towards distribution shifts. Furthermore, we introduce models trained on natural images and demonstrate that they can be used in both MRI and CT reconstruction, outperforming models trained on medical images in OOD cases. As a result of our findings, we strongly caution against simply re-using very large networks and encourage researchers to adapt the model complexity to the respective task.

Modified from Chung & Ye [1] ([Code](https://github.com/HJ-harry/score-MRI)) and Jalal et al. [2] ([Code](https://github.com/utcsilab/csgm-mri-langevin)).

## Training
Training scripts for MRI, CT and natural images are provided in ```training_scripts```.

## Evaluation
Scripts in ```evaluation``` demonstrate how the reconstructions are calculated. ```hist.sh``` creates the image statistics shown in Fig. 1.

## Cite This Work
Please consider citing the following publications underlying this work:
```
@article{Glaszner2026,
  title={Bigger Isn’t Always Better: Towards a General Prior for Medical Image Reconstruction},
  author={Glaszner, Lukas and Zach, Martin and Pock, Thomas},
  journal={International Journal of Computer Vision},
  volume={134},
  number={6},
  pages={307},
  year={2026},
  doi={10.1007/s11263-026-02904-1},
  publisher={Springer}
}

@inproceedings{Glaszner2024,
  title={Bigger Isn’t Always Better: Towards a General Prior for Medical Image Reconstruction},
  author={Glaszner, Lukas and Zach, Martin},
  booktitle={DAGM German Conference on Pattern Recognition},
  pages={275--291},
  year={2024},
  doi={10.1007/978-3-031-85181-0_18},
  publisher={Springer Nature Switzerland}
}
```


## References
[1] Chung, H., Ye, J.C.: Score-based diffusion models for accelerated MRI. Medical Image Analysis 80, 102479 (2022).

[2] Jalal, A., et al.: Robust compressed sensing MRI with deep generative priors. In: Advances in Neural Information Processing Systems. vol. 34, pp. 14938–14954. Curran Associates, Inc. (2021).
