# laion-3d
Collect large 3d dataset and build models


https://github.com/LAION-AI/project-menu/issues/23


## Dataset specs

Can be:
* 3d alone
* 3d text
* 3d image

Examples:
* https://github.com/rom1504/minecraft-schematics-dataset

## Datasets

* [30k samples from thingiverse](https://zenodo.org/record/1098527) (3d printing STL model files)
* [Fusion360Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset)
* [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download), a dataset of Amazon products with metadata, catalog images, and 3D models.
* [Large Geometric Models Archive](https://www.cc.gatech.edu/projects/large_models/about.html)
* [FaceScape](https://facescape.nju.edu.cn/Page_Download/), a large-scale detailed 3D face dataset (application required).
* [Redwood 3DScan](https://github.com/isl-org/redwood-3dscan), more than ten thousand 3D scans of real objects.
* [Human3.6M](http://vision.imar.ro/human3.6m/description.php), 3.6 million 3D human poses and corresponding images.
* [Semantic3D](https://www.semantic3d.net/), a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total.
* [SceneNN / ObjectNN](https://github.com/hkust-vgd/scenenn), an RGB-D dataset with more than 100 indoor scenes along with RGB-D objects extracted and split into 20 categories.
* [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), a dataset of 3D furnished rooms with layouts and semantics.
* [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), a dataset of 3D furniture shapes with textures.
* [ABC](https://deep-geometry.github.io/abc-dataset/0), a collection of one million Computer-Aided Design (CAD) models
* [Structured3D](https://structured3d-dataset.org/#download), a large-scale photo-realistic dataset containing 3.5K house designs with a variety of ground truth 3D structure annotations.
* [ShapeNet](https://shapenet.org/), a richly-annotated, large-scale dataset of 3D shapes.
* [FixIt!](https://drive.google.com/drive/folders/1h9kMRilQcjbD4Tyt58pmMUEnMIicNATi), a dataset that contains about 5k poorly-designed 3D physical objects paired with choices to fix them.
* [ModelNet](http://modelnet.cs.princeton.edu/#), a comprehensive clean collection of 3D CAD models for objects.

## Models

### Depth Estimation

* Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging
    * Paper: [https://arxiv.org/abs/2105.14021](https://arxiv.org/abs/2105.14021)
    * Code: [https://github.com/compphoto/BoostingMonocularDepth](https://github.com/compphoto/BoostingMonocularDepth)
    * Follow up paper: [https://arxiv.org/abs/2012.09365](https://arxiv.org/abs/2012.09365)
    * Follow up code: [https://github.com/aim-uofa/AdelaiDepth](https://github.com/aim-uofa/AdelaiDepth) (although firs code includes it)
* Self-supervised Learning of Depth Inference for Multi-view Stereo
    * Paper: [https://arxiv.org/abs/2104.02972](https://arxiv.org/abs/2104.02972)
    * Code: [https://github.com/JiayuYANG/Self-supervised-CVP-MVSNet](https://github.com/JiayuYANG/Self-supervised-CVP-MVSNet)

### Generation

* Sketch2Model - View-Aware 3D Modeling from Single Free-Hand Sketches
    * Paper: [https://arxiv.org/abs/2105.06663](https://arxiv.org/abs/2105.06663)
    * Code: [https://github.com/bennyguo/sketch2model](https://github.com/bennyguo/sketch2model)
* SceneFormer - Indoor Scene Generation with Transformers
    * Paper: [https://arxiv.org/abs/2012.09793](https://arxiv.org/abs/2012.09793)
    * Code: [https://github.com/cy94/sceneformer](https://github.com/cy94/sceneformer)
* Image2Lego - Customized LEGO Set Generation from Images
    * Paper: [https://arxiv.org/abs/2108.08477](https://arxiv.org/abs/2108.08477)
    * Code: 😥
* Neural RGB-D Surface Reconstruction
    * Paper: [https://arxiv.org/abs/2104.04532](https://arxiv.org/abs/2104.04532)
    * Code: 😥
* SP-GAN - Sphere-Guided 3D Shape Generation and Manipulation
    * Paper: [https://arxiv.org/abs/2108.04476](https://arxiv.org/abs/2108.04476)
    * Code: [https://github.com/liruihui/sp-gan](https://github.com/liruihui/sp-gan)
* Style-based Point Generator with Adversarial Rendering for Point Cloud Completion
    * Paper: [https://arxiv.org/abs/2103.02535](https://arxiv.org/abs/2103.02535)
    * Code: [https://github.com/microsoft/SpareNet](https://github.com/microsoft/SpareNet)
* Learning to Stylize Novel Views
    * Paper: [https://arxiv.org/abs/2105.13509](https://arxiv.org/abs/2105.13509)
    * Code: [https://github.com/hhsinping/stylescene](https://github.com/hhsinping/stylescene)
* RetrievalFuse - Neural 3D Scene Reconstruction with a Database
    * Paper: [https://arxiv.org/abs/2104.00024](https://arxiv.org/abs/2104.00024)
    * Code: [https://github.com/nihalsid/retrieval-fuse](https://github.com/nihalsid/retrieval-fuse)
* Geometry-Free View Synthesis - Transformers and no 3D Priors
    * Paper: [https://arxiv.org/abs/2104.07652](https://arxiv.org/abs/2104.07652)
    * Code: [https://github.com/CompVis/geometry-free-view-synthesis](https://github.com/CompVis/geometry-free-view-synthesis)
* ShapeFormer - Transformer-based Shape Completion via Sparse Representation
    * Paper: [https://arxiv.org/abs/2201.10326](https://arxiv.org/abs/2201.10326)
    * Code: 😥

### Representations (🙄)

#### SDF-based (mostly)

* Neural Parts - Learning Expressive 3D Shape Abstractions with Invertible Neural Representations
    * Paper: [https://arxiv.org/abs/2103.10429](https://arxiv.org/abs/2103.10429)
    * Code: [https://github.com/paschalidoud/neural_parts](https://github.com/paschalidoud/neural_parts)
* DeepSDF - Learning Continuous Signed Distance Functions for Shape Representation
    * Paper: [https://arxiv.org/abs/1901.05103](https://arxiv.org/abs/1901.05103)
    * Code: [https://github.com/Facebookresearch/deepsdf](https://github.com/Facebookresearch/deepsdf)
* Spline Positional Encoding for Learning 3D Implicit Signed Distance Fields
    * Paper: https://arxiv.org/abs/2106.01553
    * Code: [https://github.com/microsoft/SplinePosEnc](https://github.com/microsoft/SplinePosEnc)

#### Hashing

* Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
    * Paper: [https://arxiv.org/abs/2201.05989](https://arxiv.org/abs/2201.05989)
    * Code: [https://github.com/nvlabs/instant-ngp](https://github.com/nvlabs/instant-ngp)


#### Implicit

* Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image
    * Paper: [https://openreview.net/forum?id=U8pbd00cCWB](https://openreview.net/forum?id=U8pbd00cCWB)
    * Code: maskedURL lol
* NeuS - Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction
    * Paper: [https://arxiv.org/abs/2106.10689](https://arxiv.org/abs/2106.10689)
    * Code: [https://github.com/Totoro97/NeuS](https://github.com/Totoro97/NeuS)
* From data to functa - Your data point is a function
    * Paper: [https://arxiv.org/abs/2201.12204](https://arxiv.org/abs/2201.12204)
    * Code: 😥
    * Previous work: [https://arxiv.org/abs/2102.04776](https://arxiv.org/abs/2102.04776)
    * Code: https://github.com/EmilienDupont/neural-function-distributions
* Multiresolution Deep Implicit Functions for 3D Shape Representation
    * Paper: [https://arxiv.org/abs/2109.05591](https://arxiv.org/abs/2109.05591)
    * Code: 😥
* Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields
    * Paper: [https://arxiv.org/abs/2106.05187](https://arxiv.org/abs/2106.05187)
    * Code: [https://github.com/yifita/idf](https://github.com/yifita/idf)
* Implicit Neural Representations with Periodic Activation Functions
    * Paper: [https://arxiv.org/abs/2006.09661](https://arxiv.org/abs/2006.09661)
    * Follow up: [https://arxiv.org/abs/2104.03960](https://arxiv.org/abs/2104.03960)
    * Code: [https://github.com/lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch)
* Volume Rendering of Neural Implicit Surfaces
    * Paper: [https://arxiv.org/abs/2106.12052](https://arxiv.org/abs/2106.12052)
    * Code: [https://github.com/ventusff/neurecon](https://github.com/ventusff/neurecon)
* HyperCube - Implicit Field Representations of Voxelized 3D Models
    * Paper: [https://arxiv.org/abs/2110.05770](https://arxiv.org/abs/2110.05770)
    * Code: [https://github.com/mproszewska/hypercube](https://github.com/mproszewska/hypercube)
* Convolutional Occupancy Networks
    * Paper: [https://arxiv.org/abs/2003.04618](https://arxiv.org/abs/2003.04618)
    * Code: https://github.com/autonomousvision/convolutional_occupancy_networks

### Neural/Differential Rendering

* GANcraft - Unsupervised 3D Neural Rendering of Minecraft Worlds
    * Paper: [https://arxiv.org/abs/2104.07659](https://arxiv.org/abs/2104.07659)
    * Code: [https://github.com/NVlabs/imaginaire](https://github.com/NVlabs/imaginaire)
* ADOP: Approximate Differentiable One-Pixel Point Rendering
    * Paper: [https://arxiv.org/abs/2110.06635](https://arxiv.org/abs/2110.06635)
    * Code: [https://github.com/darglein/ADOP](https://github.com/darglein/ADOP)
    * youtube: 


### NeRF

* Editing Conditional Radiance Fields
    * Paper: [http://editnerf.csail.mit.edu/paper.pdf](http://editnerf.csail.mit.edu/paper.pdf)
    * Code: [https://github.com/stevliu/editnerf](https://github.com/stevliu/editnerf)
* GIRAFFE - Representing Scenes as Compositional Generative Neural Feature Fields
    * Paper: [https://arxiv.org/abs/2011.12100](https://arxiv.org/abs/2011.12100)
    * Code: [https://github.com/autonomousvision/giraffe](https://github.com/autonomousvision/giraffe)
* NeX - Real-time View Synthesis with Neural Basis Expansion
    * Paper: [https://arxiv.org/abs/2103.05606](https://arxiv.org/abs/2103.05606)
    * Code: [https://github.com/nex-mpi/nex-code](https://github.com/nex-mpi/nex-code)
* Putting NeRF on a Diet - Semantically Consistent Few-Shot View Synthesis
    * Paper: [https://arxiv.org/abs/2104.00677](https://arxiv.org/abs/2104.00677)
    * Code: [https://github.com/ajayjain/DietNeRF](https://github.com/ajayjain/DietNeRF)
* Unconstrained Scene Generation with Locally Conditioned Radiance Fields
    * Paper: [https://arxiv.org/abs/2104.00670](https://arxiv.org/abs/2104.00670)
    * Code: [https://github.com/apple/ml-gsn](https://github.com/apple/ml-gsn)
* Zero-Shot Text-Guided Object Generation with Dream Fields
    * Paper: [https://arxiv.org/abs/2112.01455](https://arxiv.org/abs/2112.01455)
    * Code: [https://github.com/google-research/google-research/tree/master/dreamfields](https://github.com/google-research/google-research/tree/master/dreamfields)


### Diffusion

* Diffusion Probabilistic Models for 3D Point Cloud Generation
    * Paper: [https://arxiv.org/abs/2103.01458](https://arxiv.org/abs/2103.01458)
    * Code:  [https://github.com/luost26/diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud)
* 3D Shape Generation and Completion through Point-Voxel Diffusion
    * Paper: [https://arxiv.org/abs/2104.03670](https://arxiv.org/abs/2104.03670)
    * Code: [https://github.com/alexzhou907/PVD](https://github.com/alexzhou907/PVD)
* A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion
    * Paper: [https://arxiv.org/abs/2112.03530](https://arxiv.org/abs/2112.03530)
    * Code: [https://github.com/zhaoyanglyu/point_diffusion_refinement](https://github.com/zhaoyanglyu/point_diffusion_refinement) (empty)


### GANs

* MOGAN - Morphologic-structure-aware Generative Learning from a Single Image
    * Paper: [https://arxiv.org/abs/2103.02997](https://arxiv.org/abs/2103.02997)
    * Code: https://github.com/JinshuChen/MOGAN


### Design / Practice / Creative - Related

* Indoor Scene Generation from a Collection of Semantic-Segmented Depth Images
    * Paper: [https://arxiv.org/abs/2108.09022](https://arxiv.org/abs/2108.09022)
    * Code: [https://github.com/mingjiayang/sgsdi](https://github.com/mingjiayang/sgsdi)
* ATISS - Autoregressive Transformers for Indoor Scene Synthesis
    * Paper: [https://arxiv.org/abs/2110.03675](https://arxiv.org/abs/2110.03675)
    * Code: 😥
* Computer-Aided Design as Language
    * Paper: [https://arxiv.org/abs/2105.02769](https://arxiv.org/abs/2105.02769)
    * Code: 😥
* Patch2CAD - Patchwise Embedding Learning for In-the-Wild Shape Retrieval from a Single Image
    * Paper: [https://arxiv.org/abs/2108.09368](https://arxiv.org/abs/2108.09368)
    * Code: 😥
* Modeling Artistic Workflows for Image Generation and Editing
    * Paper: [https://arxiv.org/abs/2007.07238](https://arxiv.org/abs/2007.07238)
    * Code: [https://github.com/hytseng0509/ArtEditing](https://github.com/hytseng0509/ArtEditing)


### Benchmarks

### contrastive

* (3d, text)
* (3d, image)

### Synthetic Data Generation

https://github.com/google-research/kubric from uwu1

3d -> text 

text -> 3d 

3d -> image 

image -> 3d 
