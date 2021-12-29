PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging
========

![](assets/piano_teaser.png)

PIANOLayer is a differentiable PyTorch layer that deterministically maps from pose and shape parameters to hand bone joints and vertices.
It can be integrated into any architecture as a differentiable layer to predict bone meshes for data-driven fine-grained hand bone anatomic and semantic understanding from MRI or even RGB images.

To learn about PIANO, please visit our website: https://liyuwei.cc/proj/piano

You can find the PIANO paper at: https://www.ijcai.org/proceedings/2021/0113.pdf

---

For comments or questions, please email us at: Yuwei Li (liyw@shanghaitech.edu.cn)


System Requirements:
---

Python Dependencies:
- Numpy 
- pickle
- Pytorch		 	
- Trimesh (for mesh saving)  


Getting Started:
---

Model file:
[PIANO_RIGHT_dict.pkl](assets/PIANO_RIGHT_dict.pkl)

Demo pose:
[demo_pose.pkl](assets/demo_pose.pkl)

![](assets/demo_pose.png)

> python demo.py


Acknowledgements:
---
This model and code was developped and used for the paper *PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging* for IJCAI21.
See [project page](https://liyuwei.cc/proj/piano)

It reuses part of the great code from 
[manopth](https://github.com/hassony2/manopth/blob/master/manopth) by [Yana Hasson](https://hassony2.github.io/) and 
[pytorch_HMR](https://github.com/MandyMo/pytorch_HMR) by [Zhang Xiong](https://github.com/MandyMo)!


If you find this code useful for your research, consider citing:

```
@inproceedings{li2021piano,
title={PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging},
author={Yuwei, Li and Minye, Wu and Yuyao, Zhang and Lan, Xu and Jingyi, Yu},
booktitle={Proceedings of the 30th International Joint Conference on Artificial Intelligence, {IJCAI-21}},
year={2021}
}
```