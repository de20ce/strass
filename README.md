STRASS Dehazing: Spatio-Temporal Retinex-inspired Dehazing by an averaging of Stochastic Samples
---------------

## Abstract:

In this paper, we propose a neoteric and high-efficiency single image dehazing algorithm via contrast enhancement which is called STRASS (Spatio-Temporal Retinex-Inspired by an Averaging of Stochastic Samples) dehazing, it is realized by constructing an efficient high-pass filter to process haze images and taking the influence of human vision system into account in image dehazing principles. The novel high-pass filter works by getting each pixel using RSR and computes the average of the samples. Then the low-pass filter resulting from the minimum envelope in STRESS framework has been replaced by the average of the samples. The final dehazed image is yielded after iterations of the high-pass filter. STRASS can be run directly without any machine learning. Extensive experimental results on datasets prove that STRASS surpass the state-of-the-arts. Image dehazing can be applied in the field of printing and packaging, our method is of great significance for image pre-processing before printing.

## Introduction: Our Python Implementation Details

We open source a `Python` implementation of [our work](https://www.techscience.com/jrm/v10n5/46053): `STRASS Dehazing: Spatio-Temporal Retinex-inspired Dehazing by an averaging of Stochastic Samples`. This basic `Python` version is much slower than our `C++` implementation (not released). At least, this release can give some hints to those interested in our work for non-commercial uses.

In order to run the code, use simply `poetry` the following way:
```sh
$  poetry run python strass/strass.py img_db/y16_input.png 576 output_img_db/y16_strassni5ns3_.png 5 3 1
```
In the CLI code above:
  - `strass/strass.py` is the main script path
  - `img_db/y16_input.png` is the input image path
  - `576` corresponds to the image width (check out the info to get that!)
  - `output_img_db/y16_strassni5ns3_.png` corresponds to the output image path
  - `5` corresponds to the number of iterations
  - `3` corresponds to the number of samples


The corresponding python cmd:
```sh
$ python strass/strass.py y16_output_img_db/y16_strassni5ns3_.pnginput.png 576 output_img_db/y16_strassni5ns3_.png 5 3 1
```

If you want to use another image example, you can modify the main function in `strass/strass.py` by giving the path of your own image containing a hazy scene.

## Requirements

Python 3.10+, numpy 1.26+, pillow 10.3+, poetry 1.7+


## Citation
If you use this codebase, or otherwise found our work valuable, please cite:

```
@article{yu2022strass
  title={STRASS Dehazing: Spatio-Temporal Retinex-Inspired Dehazing by an Averaging of Stochastic Samples},
  author={Zhe, Yu and Bangyong, Sun and Di, Liu and Vincent, Whannou de Dravo and Margarita, Khokhlova and Siyuan, Wu},
  journal={Journal of Renewable Materials},
  volume={10},
  number={5},
  howpublished="\url{https://www.techscience.com/jrm/v10n5/46053}",
  pages={1381-1395}
  year={2022},
}
```

The original code written in `C++` was first presented in NTIRE2020 on Non-Homogenious Haze Challenge:

```
@inproceedings{chen2021pixelated,
  title={NTIRE 2020 Challenge on NonHomogeneous Dehazing},
  author={C. O. Ancuti et al.},
  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  howpublished="\url{https://ieeexplore.ieee.org/document/9150828}",
  page={2029-2044},
  year={2020}
}

```

This work is based on two previous work on image enhancement:

```
@inproceedings{7274895,
  author={Whannou De Dravo, Vincent and Hardeberg, Jon Yngve},
  booktitle={2015 Colour and Visual Computing Symposium (CVCS)}, 
  title={Stress for dehazing}, 
  year={2015},
  howpublished="\url{https://ieeexplore.ieee.org/document/7274895}",
  volume={},
  number={},
  pages={1-6},
}

@article{yu2022strass
  title={Spatio-Temporal Retinex-Inspired Envelope with Stochastic Sampling: A Framework for Spatial Color Algorithms},
  author={Kolås, Øyvind and Farup, Ivar and Rizzi, Alessandro},
  journal={Journal of Imaging Science and Technology},
  volume={55},
  number={4},
  howpublished="url{https://library.imaging.org/jist/articles/55/4/art00011}",
  pages={}
  year={2011},
}
```

There are more: [`Multiscale approach for dehazing using the stress framework`](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/28/20/art00040) and [`An adaptive combination of dark and bright channel priors for single image dehazing`](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2457172/JIST2017JPublished.pdf?sequence=2). You can find their references in the current paper. Since the paper is based on `Retinex` theory, you can look for many other papers on literature related to both dehazing and `Retinex`. Here are few examples of such papers cited in our work: [`Enhanced variational image dehazing`](https://www.semanticscholar.org/reader/9bc5209207a8282e57f6e233779957690415d358), [`On the duality between retinex and image dehazing`](https://openaccess.thecvf.com/content_cvpr_2018/papers/Galdran_On_the_Duality_CVPR_2018_paper.pdf), [`Physical-based optimization for non-physical image dehazing methods`](https://pubmed.ncbi.nlm.nih.gov/32225542/), etc
