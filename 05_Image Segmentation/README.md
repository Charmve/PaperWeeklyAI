# Image Segmentation Paper Top10

Curated Top 10 ImgSegmentation papers, including paper code and datasets. ⚙️


[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](./CONTRIBUTING.md) 
[![Summaries](https://img.shields.io/badge/summaries-in%20issues-%2300acee.svg?style=flat)](https://github.com/Charmve/PaperWeeklyAI/issues) ![HitCount](http://hits.dwyl.com/eugeneyan/applied-ml.svg)
[![PyTorch Code](https://img.shields.io/badge/code-PyTorch-%2300acee.svg?style=flat)](https://github.com/Charmve/Semantic-Segmentation-PyTorch)
<br>

> ## 迈微导读
> 图像分割（image segmentation）技术是计算机视觉领域的重要的研究方向，近些年，图像分割技术迅猛发展，在多个视觉研究领域都有着广泛的应用。本文盘点了近20年来影响力最大的 10 篇论文。
> <br><i><b>注：</b>这里的影响力以 Web of Science 上显示的论文的引用量排序，截止时间为``2020年9月27日``。</i>

<br>

## -TOP10-   <b>Mask R-CNN</b> 
被引频次：1839

作者：Kaiming He，Georgia Gkioxari，Piotr Dollar，Ross Girshick. <br>
发布信息: 2017，16th IEEE International Conference on Computer Vision (ICCV)

论文：https://arxiv.org/abs/1703.06870<br>
代码：https://github.com/facebookresearch/Detectron

Mask R-CNN作为非常经典的实例分割（Instance segmentation）算法，在图像分割领域可谓“家喻户晓”。Mask R-CNN不仅在实例分割任务中表现优异，还是一个非常灵活的框架，可以通过增加不同的分支完成目标分类、目标检测、语义分割、实例分割、人体姿势识别等多种不同的任务。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/9ae64d9feaa76ae2ad79cf05916b9224.png"><br>
  <img src="https://img-blog.csdnimg.cn/img_convert/96e1f619c273730854c0f370a3426794.png">
</div>

<br>


## -TOP9- <b>SegNet</b>

SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

被引频次：1937

作者: Vijay Badrinarayanan，Alex Kendall，Roberto Cipolla<br>
发布信息：2015，IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

论文：https://arxiv.org/pdf/1511.00561.pdf<br>
代码：https://github.com/aizawan/segnet

SegNet是用于进行像素级别图像分割的全卷积网络。SegNet与FCN的思路较为相似，区别则在于Encoder中Pooling和Decoder的Upsampling使用的技术。Decoder进行上采样的方式是Segnet的亮点之一，SegNet主要用于场景理解应用，需要在进行inference时考虑内存的占用及分割的准确率。同时，Segnet的训练参数较少，可以用SGD进行end-to-end训练。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/be44b807c14405dffdb58aa34d853948.png"><br>
  <img src="https://img-blog.csdnimg.cn/img_convert/263f01b9fd4b04b8878e484703793adb.png"><br>
</div>
<br>


## -TOP8- DeepLab

DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

被引频次：2160

作者: Chen Liang-Chieh，Papandreou George，Kokkinos Iasonas等.<br>
发布信息：2018，IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

DeepLabv1：https://arxiv.org/pdf/1412.7062v3.pdf<br>
DeepLabv2：https://arxiv.org/pdf/1606.00915.pdf<br>
DeepLabv3：https://arxiv.org/pdf/1706.05587.pdf<br>
DeepLabv3+：https://arxiv.org/pdf/1802.02611.pdf

代码：https://github.com/tensorflow/models/tree/master/research/deeplab

DeepLab系列采用了Dilated/Atrous Convolution的方式扩展感受野，获取更多的上下文信息，避免了DCNN中重复最大池化和下采样带来的分辨率下降问题。2018年，Chen等人发布Deeplabv3+，使用编码器-解码器架构。DeepLabv3+在2012年pascal VOC挑战赛中获得89.0%的mIoU分数。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/d034d44bca50bee71d108a9698f70de9.png"><br>
  <img src="https://img-blog.csdnimg.cn/img_convert/1ba1f9886fdec067b72871607bb49701.png"><br>
  <img src="https://img-blog.csdnimg.cn/img_convert/fb7ae61ce4a0bf5dc593ae0b1ef6a4d3.png"><br>
DeepLabv3+
</div>
<br>

## -TOP7-
Contour Detection and Hierarchical Image Segmentation

被引频次：2231

作者: Arbelaez Pablo，Maire Michael，Fowlkes Charless等.<br>
发布信息：2011，IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE<br>

论文和代码：https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html<br>

Contour Detection and Hierarchical Image Segmentation通过检测轮廓来进行分割，以解决不加交互的图像分割问题，是分割领域中非常重要的一篇文章，后续很多边缘检测算法都利用了该模型。


<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/f78612e69f244920aac87f43ea1a7005.png">
</div>
<br>


## -TOP6-
Efficient graph-based image segmentation

被引频次：3302

作者：Felzenszwalb PF，Huttenlocher DP<br>
发布信息：2004，INTERNATIONAL JOURNAL OF COMPUTER VISION<br>

论文和代码：http://cs.brown.edu/people/pfelzens/segment/

Graph-Based Segmentation 是经典的图像分割算法，作者Felzenszwalb也是提出DPM算法的大牛。该算法是基于图的贪心聚类算法，实现简单。目前虽然直接用其做分割的较少，但许多算法都用它作为基石。


<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/671a8c425f540487267f6314c584574c.png">
</div>
<br>


## -TOP5-
SLIC Superpixels Compared to State-of-the-Art Superpixel Methods

被引频次：4168

作者: Radhakrishna Achanta，Appu Shaji，Kevin Smith，Aurelien Lucchi，Pascal Fua，Sabine Susstrunk.<br>
发布信息：2012，IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

论文和代码：https://ivrlwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html

SLIC 算法将K-means 算法用于超像素聚类，是一种思想简单、实现方便的算法，SLIC算法能生成紧凑、近似均匀的超像素，在运算速度，物体轮廓保持、超像素形状方面具有较高的综合评价，比较符合人们期望的分割效果。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/adc06e5f3a9237ccecb61c41b64446f4.png">
</div>
<br>



## -TOP4-
U-Net: Convolutional Networks for Biomedical Image Segmentation

被引频次：6920

作者: Ronneberger Olaf，Fischer Philipp，Brox Thomas<br>
发布信息：2015，18th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)

论文：https://arxiv.org/pdf/1505.04597.pdf<br>
代码：https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

U-Net是一种基于深度学习的图像语义分割方法，在医学图像分割领域表现尤为优异。它基于FCNs做出改进，相较于FCN多尺度信息更加丰富，同时适合超大图像分割。作者采用数据增强（data augmentation），通过使用在粗糙的3*3点阵上的随机取代向量来生成平缓的变形，解决了可获得的训练数据很少的问题。并使用加权损失（weighted loss）以解决对于同一类的连接的目标分割。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/59334386eebd747824ce30b5a1d593fd.png">
</div>
<br>



## -TOP3-
Mean shift: A robust approach toward feature space analysis

被引频次：6996

作者: Comaniciu D，Meer P<br>
发布信息：2002，IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

论文：https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf

Meanshift是基于像素聚类的代表方法之一，是一种特征空间分析方法。密度估计(Density Estimation) 和mode 搜索是Meanshift的两个核心点。对于图像数据，其分布无固定模式可循，所以密度估计必须用非参数估计，选用的是具有平滑效果的核密度估计（Kernel density estimation，KDE）。Meanshift 算法的稳定性、鲁棒性较好，有着广泛的应用。但是分割时所包含的语义信息较少，分割效果不够理想，无法有效地控制超像素的数量，且运行速度较慢，不适用于实时处理任务。

<br>


## -TOP2-
Normalized cuts and image segmentation

被引频次：8056

作者：Shi JB，Malik J<br>
发布信息：2000，IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 

论文：https://ieeexplore.ieee.org/abstract/document/1000236<br>
论文：https://pdfs.semanticscholar.org/d5d0/2b093162096005834ee22def530de6c1f7eb.pdf

NormalizedCut是基于图论的分割方法代表之一，与以往利用聚类的方法相比，更加专注于全局解的情况，并且根据图像的亮度，颜色，纹理进行划分。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/24189229d51b49835119c3a3b61bc97a.png">
</div>
<br>



## -Top1-
Fully Convolutional Networks for Semantic Segmentation

被引频次：8170

作者: Long Jonathan，Shelhamer Evan，Darrell Trevor<br>
发布信息：2015，IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

论文：https://arxiv.org/abs/1411.4038<br>
代码：https://github.com/shelhamer/fcn.berkeleyvision.org

FCN是图像分割领域里程碑式论文。作为语义分割的开山之作，FCN是当之无愧的TOP1。它提出了全卷积网络(FCN)的概念，针对语义分割训练了一个端到端，点对点的网络，它包含了三个CNN核心思想：

（1）不含全连接层(fc)的全卷积(fully conv)网络。可适应任意尺寸输入。<br>
（2）增大数据尺寸的反卷积(deconv)层。能够输出精细的结果。<br>
（3）结合不同深度层结果的跳级(skip)结构。同时确保鲁棒性和精确性。

<div align=center>
  <img src="https://img-blog.csdnimg.cn/img_convert/9d7d0be48441e6adc66b3e910fb6afad.png">
</div>
<br>
<br>

<b>参考</b><br>
[1]FCN的学习及理解（Fully Convolutional Networks for Semantic Segmentation），CSDN <br>
[2]mean shift 图像分割 (一)，CSDN <br>
[3]https://zhuanlan.zhihu.com/p/49512872 <br>
[4]图像分割—基于图的图像分割（Graph-Based Image Segmentation），CSDN <br>
[5]https://www.cnblogs.com/fourmi/p/9785377.html <br>

<br>
<b>推荐阅读</b>

（点击标题可跳转阅读）

[1] [卷积神经网络必读的40篇经典论文，包含检测/识别/分类/分割多个领域](https://mp.weixin.qq.com/s__biz=MzIxMjg1Njc3Mw==&mid=2247489304&idx=1&sn=c2c246caa8a52b6ab046a4cf87d4a892&chksm=97beedb2a0c964a4ae536fd9e108c3c21be110fb75f6d6f1cc70e520d535269c78b80e7f6e22&scene=21#wechat_redirect)

[2] [CVPR 2020最佳学生论文分享回顾：通过二叉空间分割（BSP）生成紧凑3D网格](https://mp.weixin.qq.com/s__biz=MzIxMjg1Njc3Mw==&mid=2247492045&idx=1&sn=71473c0347cad7c396726cca2df9d650&chksm=97bd1b67a0ca9271b589b0ddbdf97417d700488f7da21b5709f13261a023395f3f595b9f2e1a&scene=21#wechat_redirect)

[3] [开源！5 行代码，快速实现图像分割，代码逐行详解，手把手教你处理图像](https://mp.weixin.qq.com/s__biz=MzIxMjg1Njc3Mw==&mid=2247488515&idx=3&sn=14ee60f5ef1d47f2c704f976691ecea4&chksm=97beeea9a0c967bfafe44b2cbbf02f5b642e5edc9707e4788bb5a4857fa01d5dd2c9852a6514&scene=21#wechat_redirect)

[4] [何恺明团队新作：图像分割精细度空前，边缘自带抗锯齿，算力仅需Mask R-CNN的2.6%](https://mp.weixin.qq.com/s__biz=MzIxMjg1Njc3Mw==&mid=2247484315&idx=1&sn=e350d800d77a8d98212c1b6725b6c044&chksm=97bef931a0c970272103b85f5016ccbf962a3eb2daf71242d7d3e2d7581c0b1a7fcc9ff349f6&scene=21#wechat_redirect)

[5] [滑动窗口也能用于实例分割，陈鑫磊、何恺明等人提出图像分割新范式](https://mp.weixin.qq.com/s__biz=MzIxMjg1Njc3Mw==&mid=2247499036&idx=1&sn=9f5bb1887c46937cef0eec39438e5ff3&chksm=97bd07b6a0ca8ea04e7dd18bfefd9216183aa2e922e04691577f61d8f1a96130745b2647c0fc&token=352776378&lang=zh_CN#rd)

<br>
* 推荐个人Github Repo：表面缺陷检测数据集<br>

<table>
        <tr>   
		<td><font size="3"><b>1.</b></font></td>&nbsp;&nbsp;
		<td><center><img width="260" src="https://github.com/Charmve/Charmve.github.io/blob/master/img/surface_dataset.png" alt="Surface Defect Detection"></center></td>
		<td>
			<font size="4">
                            <b>   Surface Defect Detection: Paper & dataset</b>
			</font>	    
			<font size="3">
			<br>
			<br>Constantly summarizing open source dataset and important critical papers in the field of surface defect research are very important.
			<br><br> 🐎📈 Constantly summarizing open source dataset and important critical papers in the field of surface defect research are very important. 🐋 <br>
        NEU-CLS, elpv-dataset, KolektorSDD, DeepPCB, AITEX, DAGM 2007, Cracks on the surface of the construction, Magnetic Tile, RSDDs Kylberg Texture, etc.
                        <ul class="list-inline">
                            <a class="github-button"
                                href="https://github.com/Charmve/Surface-Defect-Detection"
                                data-icon="octicon-star" data-show-count="true"
                                aria-label="Star Charmve/Surface-Defect-Detection on GitHub">Star</a>
                            <a class="github-button"
                                href="https://github.com/Charmve/Surface-Defect-Detection/fork"
                                data-icon="octicon-repo-forked" data-show-count="true"
                                aria-label="Fork Charmve/Surface-Defect-Detection on GitHub">Fork</a>
                       </ul>
		       <br><img src="https://img.icons8.com/material-sharp/24/000000/github.png" alt="Github" width="22px"/>
			    <a href="https://github.com/Charmve/Surface-Defect-Detection" target="_blank">https://github.com/Charmve/Surface-Defect-Detection</a>
		       <br><br>
		</td>
	</tr>
	<tr>   
		<td><font size="3"><b>2.</b></font></td>&nbsp;&nbsp;
		<td><center><img width="260" src="https://github.com/Charmve/Charmve.github.io/blob/master/img/glrass_detection.jpg" alt="GlassDetection"></center></td>
		<td>
			<font size="4">
			    <b>   Mirror&Glass Detection: Paper & dataset </b>
			</font>	    
			<font size="3">
		            <br>
			    <br>   I'm more interested in detecting general glass surfaces that may not possess any special properties, 
				and am developing computational models for automatic detection and segmentation of mirror and transparent glass surfaces.
			    <br>
				Thanks for <i>cs.cityu.edu.hk.</i>
			    <br>
			    <ul class="list-inline">
                                 <a class="github-button"
                                     href="https://github.com/Charmve/Mirror-Glass-Detection"
                                     data-icon="octicon-star" data-show-count="true"
                                     aria-label="Star Charmve/Mirror-Glass-Detection on GitHub">Star</a>
                                 <a class="github-button"
                                     href="https://github.com/Charmve/Mirror-Glass-Detection/fork"
                                     data-icon="octicon-repo-forked" data-show-count="true"
                                     aria-label="Fork Charmve/Mirror-Glass-Detection on GitHub">Fork</a>
                            </ul>
			    <br><img src="https://img.icons8.com/material-sharp/24/000000/github.png" alt="Github" width="22px"/>
				<a href="https://github.com/Charmve/Mirror-Glass-Detection" target="_blank">https://github.com/Charmve/Mirror-Glass-Detection</a>
			    <br><br>
			</font>
		</td>
	</tr>
	<tr>   
		<td><font size="3"><b>3.</b></font></td>&nbsp;&nbsp;
		<td><center><img width="260" src="https://github.com/Charmve/Charmve.github.io/blob/master/img/sne-roadseg.png" alt="SNE-RoadSeg2"></center></td>
		<td>
			<font size="4">
			    <b>   SNE-RoadSeg2: Accurate Freespace Detection</b>
			</font>	    
			<font size="3">
			    <br>
			    <br>   PyTorch implementation of SNE-RoadSeg: Incorporating Surface Normal Information into Semantic Segmentation for Accurate Freespace Detection <a href="http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750341.pdf" target="_blank">[Paper]</a>
			    <br>
                            <ul class="list-inline">
                                <a class="github-button"
                                    href="https://github.com/Charmve/SNE-RoadSeg2"
                                    data-icon="octicon-star" data-show-count="true"
                                    aria-label="Star Charmve/SNE-RoadSeg2 on GitHub">Star</a>
                                <a class="github-button"
                                    href="https://github.com/Charmve/SNE-RoadSeg2/fork"
                                    data-icon="octicon-repo-forked" data-show-count="true"
                                    aria-label="Fork Charmve/SNE-RoadSeg2 on GitHub">Fork</a>
                            </ul>
			    <br><img src="https://img.icons8.com/material-sharp/24/000000/github.png" alt="Github" width="22px"/>
				<a href="https://github.com/Charmve/SNE-RoadSeg2" target="_blank">https://github.com/Charmve/SNE-RoadSeg2</a>
			    <br><br>
			</font>
		</td>
	</tr>
</table>



<div align=center><img src="https://github.com/Charmve/PaperWeeklyAI/raw/master/MaiweiAI-com.png?raw=true" height="330" width="330"></div>

<div align=center size = 3><b>△微信扫一扫，关注我</b></div>

<br>
