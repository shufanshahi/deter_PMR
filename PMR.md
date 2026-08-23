

PMR:PrototypicalModalRebalance for Multimodal Learning
## Yunfeng Fan
## 1
## , Wenchao Xu
## 1
## , Haozhao Wang
## 2,
## *
## , Junxiao Wang
## 3,4
, and Song Guo
## 1
## 1
## The Hong Kong Polytechnic University,
## 2
Huazhong University of Science and Technology
## 3
## KAUST,
## 4
SDAIA-KAUST AI Center
## {yunfeng.fan,wenchao.xu}@polyu.edu.hk,hz
wang@hust.edu.cn,
junxiao.wang@kaust.edu.sa,song.guo@polyu.edu.hk
## Abstract
Multimodal learning (MML) aims to jointly exploit the
common  priors  of  different  modalities  to  compensate  for
their  inherent  limitations.   However,  existing  MML  meth-
ods often optimize a uniform objective for different modal-
ities, leading to the notorious “modality imbalance” prob-
lem and counterproductive MML performance.  To address
the problem, some existing methods modulate the learning
pace based on the fused modality, which is dominated by the
better modality and eventually results in a limited improve-
ment  on  the  worse  modal.   To  better  exploit  the  features
of  multimodal,  we  propose  Prototypical  Modality  Rebal-
ance (PMR) to perform stimulation on the particular slow-
learning modality without interference from other modali-
ties. Specifically, we introduce the prototypes that represent
general features for each class, to build the non-parametric
classifiers for uni-modal performance evaluation. Then, we
try to accelerate the slow-learning modality by enhancing
its clustering toward prototypes.  Furthermore, to alleviate
the suppression from the dominant modality, we introduce
a  prototype-based  entropy  regularization  term  during  the
early training stage to prevent premature convergence. Be-
sides, our method only relies on the representations of each
modality and without restrictions from model structures and
fusion methods, making it with great application potential
for various scenarios. The source code is available here
## 1
## .
## 1. Introduction
Multimodal  learning  (MML)  [24, 34, 36]  emerges  to
mimic the way humans perceive the world, i.e., from multi-
ple sense channels toward a common phenomenon for bet-
ter understanding the external environment, which has at-
tracted extensive attention in various scenarios, e.g.  video
classification  [13, 28, 37],  event  localization  [38, 43],  ac-
## *
Corresponding author
## 1
https://github.com/fanyunfeng-bit/Modal-Imbalance-PMR
direction disturbed
by dominant modal
direction to
prototypes
prototypes
update
## Interfered
by others:
## Ours:
Figure 1. The slow-learning modal’s updating direction is severely
disturbed by the dominant one, making it hard to exploit its fea-
tures. We propose to use the prototypes, the centroids of each class
in representation space, to adjust updating direction for better uni-
modal performance.  Other modalities will not interfere with the
new direction, which ensures improvement.
tion  recognition  [10, 33],  audiovisual  speech  recognition
[23, 25].  By employing a complementary manner of mul-
timodal training, it is expected that MML can achieve bet-
ter performance than using a single modality. However, the
heterogeneity of multimodal data poses challenges on how
to learn multimodal correlations and complementarities.
According to recent research [29, 42], although the over-
all  performance  of  multimodal  learning  exceeds  that  of
single-modal  learning,  the  performance  of  each  modality
tends to be far from their upper bound.  The reason behind
this phenomenon is the “modality imbalance” problem, in
which  the  dominant  modality  will  hinder  the  full  utiliza-
tion of multimodal.  The researchers [40] also claimed that
different modalities overfit and converge at different rates,
meaning  that  optimizing  the  same  objective  for  different
modalities leads to inconsistent learning efficiency.
Several methods [29, 40, 44] have been proposed to ad-
dress  the  problem.   Some  of  them  [29, 44]  try  to  modu-
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
## 20029

late the learning paces of different modalities based on the
fusion modal.  However, we find out through experiments
that the dominant modality not only suppresses the learning
rates of other modalities [29] but also interferes with their
update direction, which makes it hard to improve the per-
formance of slow-learning modalities.  Moreover, existing
methods either inevitably bring additional model structures
[5, 40] or are limited by specific fusion methods [29, 42],
which limit their application scenarios.
To tackle their limitations, we propose thePrototypical
ModalRebalance  (PMR)  strategy  to  stimulate  the  slow-
learning modality via promoting the exploitation of features
and alleviate the suppression from the dominant modality
by slowing down itself in the early training stage.
Concretely, we introduce the prototypes for each modal-
ity, which are defined as “representative embeddings of in-
stances of a class”.  We utilize the prototypes to construct
non-parametric classifiers by comparing the distances be-
tween each sample with all the prototypes to evaluate the
performance of each modality and design a new prototype-
based metric inspired by [29] to monitor the modality im-
balance degree during the training process.  Then, we pro-
pose the prototypical cross-entropy (PCE) loss to acceler-
ate the slow-learning modality by enhancing its clustering
process, as illustrated in Fig. 1.  The PCE loss can achieve
comparable performance to the cross-entropy (CE) loss [1]
in the classification task, and more importantly, it is not af-
fected by the dominant modality and gives internal impetus
for full feature exploitation instead of going on in the dis-
turbed direction.  In addition,  we introduce a prototypical
entropy  regularization  (PER)  term,  which  can  be  seen  as
a penalty on the dominant modality to prevent premature
convergence for suppression effect alleviation. Our method
only relies on the representations of each modality and with-
out restrictions from model structures and fusion methods.
Therefore, the PMR strategy has great generality potential.
To  summarize,  our  contributions  in  this  paper  are  as  fol-
lows:
-  We analyze the modal imbalance problem and find that
during the training process, the deviation of the gradi-
ent update direction of the uni-modal became larger,
indicating that we should not regulate along the origi-
nal gradient.
-  We  propose  PMR  to  address  the  modal  imbalance
problem by actively accelerating slow-learning modal-
ities with PCE loss and simultaneously alleviating the
suppression of the dominant modality via PER.
-  We conduct comprehensive experiments and demon-
strate that 1) PMR can achieve considerable improve-
ments over existing methods; 2) PMR is independent
of the fusion method or model structure and has strong
advantages in generality.
## 2. Related Works
2.1. Imbalanced multimodal learning
Several recent studies [5, 40, 41] have shown that many
multimodal DNNs cannot achieve better performance com-
pared  to  the  best  single-modal  DNNs.   Wanget  al.  [40]
found  that  different  modalities  overfit  and  generalize  at
different rates and thus obtain suboptimal solutions when
jointly training them using a unified optimization strategy.
Penget al. [29] proposed that the better-performing modal-
ity will dominate the gradient update while suppressing the
learning process of the other modality.  Furthermore, it has
been reported that multimodal DNNs can exploit modal bias
in the data, which is inconsistent with the expectation of ex-
ploiting cross-modal interactions in VQA [8, 12, 41].
Several approaches have been proposed recently to deal
with  the  modal  imbalance  problem  [5, 29, 40, 44].   Wang
et al. [40] used additional classifiers for each modality and
its fusion modality and then optimized the gradient mixing
problem they introduced to obtain better weights for each
branch.   Duet  al.  [5]  tried  to  improve  uni-modal  perfor-
mance  by  distilling  knowledge  from  well-trained  models.
However, these approaches introduce more model structure
and computational effort, which makes the training process
more  complex  and  expensive.   Xiaoet  al.  [44]  proposed
DropPathway,  which  randomly  drops  the  audio  pathway
during training as a regularization technique to adjust the
learning paces between visual and audio pathways. Penget
al. [29] chose to slow down the learning rate of the mighty
modality by online modulation to lessen the inhibitory ef-
fect on the other modality. Although a certain degree of im-
provement is achieved, such approaches do not impose the
intrinsic  motivation  of  improvement  on  the  slow-learning
modality, making the improvement of this modality a pas-
sive rather than an active behavior. Besides, the interference
from other modalities will hinder the improvement by mod-
ulation based on the fused modality data.  Furthermore, the
application scenarios of these methods are limited by fusion
methods or model structures. In this paper, we aim to power
the slow-learning modality to reduce the gap with the domi-
nant one and allow it to be used in conjunction with various
fusion methods and model architectures.
2.2. Prototypical network
Prototypical networks were originally proposed to solve
few-shot or zero-shot classification problems [4, 7, 32, 35],
based  on  the  idea  that  there  is  an  embedding,  defined  as
a prototype, which is surrounded by points from the same
class.  Recently, this approach has been widely used to ad-
dress long-tail recognition [45], domain adaptation [3, 27],
and facilitate unsupervised learning [20], since prototypes
can  be  used  to  represent  general  features  of  a  class.   In
[20, 35], prototypes were interpreted as non-parametric pro-
## 20030

## 0255075100
## Epoch
## 10
## 20
## 30
## 40
## 50
## 60
s
## Fusion
## Audio
## Visual
## (a)
## 0255075
## Epoch
## 0.40
## 0.45
## 0.50
## 0.55
## Accuracy
## Baseline
## OGM-GE
## Acc
## (b)
## 0255075
## Epoch
## 30
## 40
## 50
## 60
## 70
## Angle
Audio angle
visual angle
## (c)
Figure 2.  (a) Performance of each modality and their fusion.  (b) Training accuracy of multimodal with different modulations.  Baseline
means no extra modulation.  Acc increases the gradient magnitude of the slow-learning modality.  OMG-GE [29] reduces the gradient
magnitude of better modality.  (c) The gradient direction discrepancy (angle) between uni-modal and multimodal on the baseline.  The
results were acquired from CREMA-D.
totypical classifiers that perform on par or even better than
parametric linear classifiers. In this paper, we leverage pro-
totypes  to  build  non-parametric  classifiers  to  evaluate  the
performance of each modality feature.
## 3. Modality Imbalance Analysis
Without loss of generality, we consider two input modal-
ities asm
## 0
andm
## 1
.   The dataset is denoted asD,  which
consists of instances and their corresponding labels(x,y),
wherex= (x
m
## 0
## ,x
m
## 1
## ,y) ={x
m
## 0
i
## ,x
m
## 1
i
## ,y
i
## }
i=1,2,...,N
## ,
y={1,2,...,M}, andMis the number of categories . The
goal is to train a model with this data to predictyfromx.
We use a multimodal DNN with two uni-modal branches
for  prediction.   Each  branch  has  an  encoder,  denoted  as
φ
## 0
## ,φ
## 1
,  to  extract  features  of  respective  modal  datax
m
## 0
andx
m
## 1
.   The representation outputs of encoders are de-
noted asz
## 0
## =φ
## 0
##  
θ
## 0
## ,x
m
## 0
## 
andz
## 1
## =φ
## 1
##  
θ
## 1
## ,x
m
## 1
## 
, where
θ
## 0
andθ
## 1
are  the  parameters  of  encoders.   The  two  uni-
modal encoders are connected through the representations
by some kind of fusion, which is prevalent in multimodal
learning [19, 22, 46].  In this work, we have tried some dif-
ferent fusion methods. For simplicity, we use[·,·]to denote
fusion operation.  LetW∈R
## M×
## (
d
z
## 0
## +d
z
## 1
## )
andb∈R
## M
denote the parameters of the linear classifier to produce the
logits output:
f(x) =W
## 
φ
## 0
##  
θ
## 0
## ,x
m
## 0
## 
## ;φ
## 1
##  
θ
## 1
## ,x
m
## 1
## 
## +b(1)
The cross-entropy loss of the multimodal model is
## L
## CE
## =−
## 1
## N
## X
## N
i=1
log
e
f(x
i
## )
y
i
## P
## M
k=1
e
f(x
i
## )
k
## (2)
The gradient of the softmax logits output with true label
y
i
should be:
## ∂L
## CE
## ∂f(x
i
## )
y
i
## =
e
## (
## W
## [
φ
## 0
## ;φ
## 1
## ]
## +b
## )
y
i
## P
## M
k=1
e
(W[φ
## 0
## ;φ
## 1
## ]+b)
k
## −1(3)
For    convenience,we    simplifyφ
## 0
##  
θ
## 0
## ,x
m
## 0
## 
and
φ
## 1
##  
θ
## 1
## ,x
m
## 1
## 
asφ
## 0
andφ
## 1
.    According  to  Eq.  (3),  the
final  gradient  is  influenced  by  the  performance  of  the
fused  modality.    However,  we  cannot  directly  judge  the
contribution  of  the  two  modalities.   To  do  it,  we  take  a
simple fusion method, summation, as the example here:
f(x) =W
## 0
## ·φ
## 0
##  
θ
## 0
## ,x
m
## 0
## 
## +b
## 0
## +W
## 1
## ·φ
## 1
##  
θ
## 1
## ,x
m
## 1
## 
## +b
## 1
## (4)
whereW
## 0
## ∈R
## M×d
z
## 0
## ,W
## 1
## ∈R
## M×d
z
## 1
andb
## 0
## ,b
## 1
## ∈R
## M
are the parameters for individual modal classifier.   There-
fore, we use the logits output of the ground truth as the per-
formance for each modality and their summation fusion:
s
## 0
= softmax
##  
## W
## 0
## ·φ
## 0
## +b
## 0
## 
y
s
## 1
= softmax
##  
## W
## 1
## ·φ
## 1
## +b
## 1
## 
y
s
fu
= softmax
##  
## W
## 0
## ·φ
## 0
## +b
## 0
## +W
## 1
## ·φ
## 1
## +b
## 1
## 
y
## (5)
As shown in Fig. 2a, the performance of the audio modal
is very similar to the multimodal during training and the vi-
sual modal is much worse in CREMA-D [2], which means
better modality contributes more to the gradient because of
higher performance similarity with their fusion.  Moreover,
the visual modal is severely suppressed in the multimodal
learning process because of the excessive dominance of gra-
dient updates by the audio modal.  Therefore,  we have to
mitigate the inhibition on visual modal to fully exploit vi-
sual features. A straightforward approach can be to increase
the magnitude of the gradient.  We test a similar way with
OGM-GE [29], named Acc, to increase the gradient magni-
tude of the slow-learning modality instead of lowering the
gradient magnitude of the better modality in OGM-GE. The
results are shown in Figs. 2b and 2c.
As demonstrated in Fig. 2b, increasing the gradient mag-
nitude of the slow-learning modality (visual) could improve
the validation accuracy a little bit but not as obviously as
## 20031

MutimodalData as
## Input
## Encoder0
## Encoder1
## Representation
## Fusion & Classifier
## Prototypes
PCE Loss
CE Loss
PCE Loss
## Backward
## PER
## PER
Figure 3. The pipeline of modality modulation with prototypical modal rebalance strategy.
OGM-GE does. To find the reason for the phenomenon, we
use the uni-modal outputs
## 0
ands
## 1
to calculate the gradient
for each modal branch additionally with CE loss and illus-
trate  the  direction  discrepancy  of  gradients  between  each
uni-modal and multimodals
fu
, as shown in Fig. 2c.  The
angle  between  the  actual  gradient  update  direction  (from
the multimodal output) and each modal’s guidance direc-
tion (from the uni-modal output) increases dramatically dur-
ing training, in the meantime, remaining acute.  Therefore,
the two modalities influence each other, resulting in a larger
gap between the gradient update direction obtained by the
fused modality and the expected direction of each modal-
ity. That means the slow-learning modality cannot fully ex-
ploit its feature with the disturbance from other modalities,
ultimately making the method of modulating the gradient
amplitude limited, as illustrated in Fig. 1.
## 4. Prototypical Modal Rebalance
4.1. Prototypical CE loss for modal acceleration
As  discussed  above,  due  to  inconsistency  in  perfor-
mance between modalities,  they will affect each other on
the exploitation of self-feature in multimodal learning, and
the  modality  with  worse  performance  is  particularly  sup-
pressed, which limits multimodal performance.  In order to
solve the problem, we aim to facilitate the exploitation of
the features of the slow-learning modality via prototypes,
as shown in Fig. 3.
The  performance  estimation  in  Eq.  (5)  needs  the  log-
its output after the classifier and of course, the logits have
to  be  decomposable  into  separate  components  of  the  two
modalities.   However,  this  constraint  is  too  strong,  limit-
ing  the  estimation’s  application  in  most  scenarios.   Aim-
ing  to  implement  a  universal  estimation  method,  we  in-
troduce  the  prototypes  of  the  categories.    With  the  data
x={x
m
## 0
i
## ,x
m
## 1
i
## ,y
i
## }
i=1,2,...,N
, we could produce the rep-
resentationsz=
## 
z
## 0
i
## ,z
## 1
i

i=1,2,...,N
in the training process.
We denotez
## 0
k
## =
## 
z
## 0
k
i

i=1,2,...,N
k
## ,z
## 1
k
## =
## 
z
## 1
k
i

i=1,2,...,N
k
as the subset data of each category, whereN
k
is the num-
ber of classk, andk= 1,2,···,M.  The prototype is the
centroid of each category data:
c
## 0
k
## =
## 1
## N
k
## X
## N
k
i=1
z
## 0
k
i
c
## 1
k
## =
## 1
## N
k
## X
## N
k
i=1
z
## 1
k
i
## (6)
Then, we use prototypes to produce a distribution over
classes for the input dataxbased on a softmax over dis-
tances to the prototypes in the embedding space for each
modality as in [35]:
p
φ
## 0
i
## (y=k|x
m
## 0
i
## ) =
exp
##  
## −d
##  
φ
## 0
## (x
m
## 0
i
## ),c
## 0
k
## 
## P
k
## ′
exp (−d(φ
## 0
## (x
m
## 0
i
## ),c
## 0
k
## ′
## ))
p
φ
## 1
i
## (y=k|x
m
## 1
i
## ) =
exp
##  
## −d
##  
φ
## 1
## (x
m
## 1
i
## ),c
## 1
k
## 
## P
k
## ′
exp (−d(φ
## 1
## (x
m
## 1
i
## ),c
## 1
k
## ′
## ))
## (7)
whered(·,·)is the distance function, which is the Euclidean
distance in this paper. Here, we design the imbalanced ratio
inspired by [29]:
ρ
t
## =
## P
i∈B
## 0
t
p
φ
## 0
i
## P
i∈B
## 1
t
p
φ
## 1
i
## (8)
## B
## 0
t
andB
## 1
t
are the batch data at training step timet, there-
fore,  we could use it to evaluate the degree of imbalance
between two modalities in real-time.  The metric is calcu-
lated using only representations and prototypes, which are
both computationally independent of the fusion method and
classifier structure.
## 20032

According to Sec. 3, in order to exploit more informa-
tion of the slow-learning modality, increasing the gradient
magnitude of the suppressed modal is not an ideal solution
due to the perturbation by another modality.  We leverage
the prototypes to introduce the PCE loss that is independent
of another modality to promote the slow-learning modal’s
performance:
## L
## 0
## PCE
(f) =E
p
## (
x
m
## 0
i
## ,y
## )
## "
## −log
exp
##  
## −d
##  
z
## 0
i
## ,c
## 0
y
## 
## P
## M
k=1
exp (−d(z
## 0
i
## ,c
## 0
k
## ))
## #
## L
## 1
## PCE
(f) =E
p
## (
x
m
## 1
i
## ,y
## )
## "
## −log
exp
##  
## −d
##  
z
## 1
i
## ,c
## 1
y
## 
## P
## M
k=1
exp (−d(z
## 1
i
## ,c
## 1
k
## ))
## #
## (9)
The acceleration loss is the weighted combination of CE
loss and PCE loss:
## L
acc
## =L
## CE
+α·βL
## 0
## PCE
+α·γL
## 1
## PCE
## (10)
whereαis a hyper-parameter to control the degree of mod-
ulation.  Withρ
t
to evaluate the imbalance degree dynam-
ically,  we  are  able  to  regulate  the  learning  speed  of  each
modality by adjusting the coefficientsβandγin a simple
way:
## (
β=clip
## 
## 0,
## 1
ρ
t
## −1,1
## 
## ,γ= 0ρ
t
## <1
β= 0,γ=clip(0,ρ
t
## −1,1)ρ
t
## ⩾1
## (11)
whereclip(a,b,c)is the truncate function which constrains
bto be betweenaandc.  In this way, the slower-learning
modality  can  be  facilitated  to  exploit  its  features,  while
the better modality maintains the original learning strategy,
which  mitigates  the  modal  imbalance  phenomenon.   Be-
cause  the  PCE  loss  only  uses  the  representation  of  each
modal  encoder,  our  method  can  be  applied  to  any  modal
fusion scenario, as long as the model itself has encoder(s)
to extract features for two modalities respectively. Besides,
to  stabilize the  learning process  and  reduce the  computa-
tion cost, we compute the prototypes based on a subset of
training data in a momentum fashion between each training
epoch:
c
## 0
k
## |
old
## =εc
## 0
k
## |
old
## + (1−ε)c
## 0
k
## |
new
c
## 1
k
## |
old
## =εc
## 1
k
## |
old
## + (1−ε)c
## 1
k
## |
new
## (12)
wherec
k
## |
old
is  the  previous  prototype  in  last  epoch  and
c
k
## |
new
is the prototype calculated in the current epoch.
4.2. Prototypical entropy regularization for inhibi-
tion reduction
With the help of PCE loss, we could accelerate the slow-
learning modality.   Nevertheless,  the fusion output is still
a  hindrance  that  restrains  the  improvement  of  the  slow-
learning  modality.   As  demonstrated  in  Figs.  2a  and  2c,
the distraction from other modalities increases dramatically
Algorithm 1:multimodal learning with PMR.
Input:Input dataD={x
m
## 0
i
## ,x
m
## 1
i
## ,y
i
## }
i=1,2,...,N
## ,
subsetD
## ∫
, initialized model parametersθ
## 0
## ,
θ
## 1
, hyper-parametersα,ε, epoch numberE,
regularization epoch numberE
r
## .
int e=0;
whilee < Edo
Obtain the subset representationsz
## 0
s
## ,z
## 1
s
by
feeding-forwardD
## ∫
to the model;
Calculate the detached prototypesc
## 0
## ,c
## 1
using
## Eq. (12);
foreachmini-batch dataB
t
inDat steptdo
Obtain the representationsz
## 0
t
## ,z
## 1
t
by
feeding-forwardB
t
to the model;
## Calculateρ
t
using Eq. (8);
ife < E
r
then
Calculate the final lossL
final
using
Eqs. (10), (11) and (13);
else
Calculate the final loss using Eq. (10)
Update the model based onL
final
## ;
e=e+1;
when the performance gap between modalities is quite dis-
tinct. In order to reduce the inhibition, we propose the pro-
totypical entropy regularization (PER) terms to slow down
the convergence speed of the dominant modality:
## L
final
## =L
acc
−μ·γH
## 0
##  
π
##  
## −d
##  
z
## 0
## ,c
## 0
y
## 
−μ·βH
## 1
##  
π
##  
## −d
##  
z
## 1
## ,c
## 1
y
## 
## (13)
whereπis thesoftmaxfunction which produces the prob-
abilities andHis the entropy.μis a hyper-parameter.  The
coefficientsβandγare multiplied on the opposite modality
compared  with  Eq.  (10),  which  means  accelerating  slow-
learning modality and preventing modality premature con-
vergence happens at the same time.  One point worth em-
phasizing is that we only add the regularization term in the
first training few epochs to reduce the inhibitory effect in
the early training stage to avoid performance damage of this
modal. Overall, the pseudo-code of PMR is provided in Al-
gorithm 1.
## 5. Evaluation
## 5.1. Datasets
CREMA-D[2] is an audio-visual dataset for the study
of emotion recognition, which consists of facial and vocal
emotional expressions. The emotional states can be divided
into 6 categories:happy,sad,anger,fear,disgustandneu-
tral.  There is a total of 7442 clips in the dataset, which are
## 20033

randomly divided into 6698 samples as the training set and
744 samples as the testing set.
AVE[38]  is  an  audio-visual  video  dataset  for  audio-
visual event localization, in which there are 28 event classes
and 4,143 10-second videos with both auditory and visual
tracks as well as second-level annotations.  All the videos
are collected from YouTube. In our experiments, we extract
the frames from event-localized video segments and capture
the audio clips within the same segments, constructing a la-
beled  multimodal  classification  dataset.   The  training  and
validation split of the dataset follows [38].
Colored-and-gray MNIST[16]  is  a  synthetic  dataset
based  on  MNIST  [18],  which  is  denoted  as  CG-MNIST.
Each instance contains two kinds of images,  a gray-scale
and  a  monochromatic  image.    In  the  training  set,  there
are  60,000  instances  and  the  monochromatic  images  are
strongly color-correlated with their digit labels.  In the val-
idation  set,  the  number  of  instances  is  10,000,  while  the
monochromatic  images  are  weakly  color-correlated  with
their labels.  In this work, we consider the monochromatic
image as the first modality and the gray-scale image as the
second modality, as in [42].  This dataset is used to prove
the method’s effectiveness beyond audio-visual.
5.2. Experimental settings
For   datasets   CREMA-D   and   AVE,   we   adopt   the
ResNet18  [9]  as  our  encoder  backbones  and  map  the  in-
put data as 512-dimensional vectors.   For audio modality,
the data is converted to a spectrogram of size 257×1,004 for
AVE and 257x299 for CREMA-D. For visual modality, we
extract 10 frames from the video clips and randomly select
3 frames for CREMA-D (4 for AVE) to build the training
dataset.   For Colored-and-gray MNIST, we build a neural
network with 4 convolution layers and 1 average pool layer
as the encoder. We train all the models with mini-batch size
64 and an SGD optimizer [31] with a momentum of 0.9 and
a weight decay of 1e-4.  The learning rate is initialized as
1e-3 and gradually decays to 1e-4.  The subset of data for
prototype calculation is one-tenth the size of training data
and is also extracted from the training data.αis set to 1
or 2, for different datasets.μis set to a small value 1e-2
or 1e-3 for different datasets.  All of our experiments were
performed on one NVIDIA GeForce RTX 3090 GPU.
5.3. Effectiveness on the multimodal task
Comparison on conventional fusion methods.In this ex-
periment, we apply the PMR strategy on 4 kinds of basic
fusion methods:  concatenation [26], summation, film [30]
and  gated  [15].   Among  these,  summation  is  the  type  of
late  fusion  and  the  other  three  belong  to  intermediate  fu-
sion [11, 21] method.  The logit output of summation and
concatenation can be split into two individual parts for each
modality  combined  with  a  linear  classifier,  as  discussed
DatasetCREMA-DAVECG-MNIST
MethodAccAccAcc
## Uni-modal154.462.199.3
## Uni-modal2
## 58.031.060.4
## Concatenation53.265.458.4
## Summation52.965.159.1
## Film57.264.460.0
## Gated58.463.559.8
## Concatenation†61.167.177.2
## Summation†59.468.178.5
## Film†61.866.476.3
## Gated†59.962.768.7
Table 1. Performance on CREMA-D, AVE and Colored-and-gray
MNIST  dataset  with  various  fusion  methods.†indicates  PMR
strategy is applied. PMR gets great performance improvement on
nearly all scenarios
in  [29].   In  film  and  gated,  features  between  the  modali-
ties are fused in more complicated ways therefore the logit
output cannot be fully split. The results are shown in Tab. 1,
which also includes the performance trained on every sin-
gle  modality.   Modal1  is  audio  and  modal2  is  visual  for
CREMA-D and AVE, while modal1 is gray-modality and
modal2 is colored-modality for CG-MNIST. According to
the results, we can find that the performance of each uni-
modal model in each dataset is inconsistent,  as the audio
performance is worse than the visual in CREMA-D, and on
the contrary, the audio performance is better than the visual
in AVE. In  addition,  uni-modal performance may  outper-
form the vanilla fusion methods.  For example, uni-visual
performance is obviously better than the concatenation and
summation  for  CREMA-D,  so  the  uni-gray  performance
is in CG-MNIST, indicating the inhibitory relationship be-
tween modalities. We get a significant improvement on the
three  datasets  compared  with  each  vanilla  fusion  method
when our PMR strategy is applied, except for a slight de-
crease in gated on the AVE dataset.
Improved performance compared with other baselines.
We  compare  our  PMR  strategy  with  other  three  modula-
tion strategies for modality imbalance: Modality-Drop [44],
Gradient-Blending [40] and OGM-GE [29].   We compare
them with concatenation and film fusion methods.  All re-
sults are shown in Tab. 2. We can notice that all the modula-
tion methods achieve better performance compared with the
baseline and our PMR gets the best among them. The main
improvement contribution comes from PCE loss to acceler-
ate the slow-learning modality while combining with PER
will still get almost a 1% increase or at least stay the same.
Apart from these, we also note that Gradient-Blending has
to train additional uni-modal classifiers and requires the val-
idation results with extra computation, while OGM-GE can
only be directly used in concatenation but not film, there-
fore, we modulate the training process every few epochs as
## 20034

DatasetCREMA-DAVE
FusionConcatFilmConcatFilm
## Naive53.257.265.464.4
## Modality-drop55.858.366.463.9
Grad-Blending
## 56.758.765.565.2
## OGM-GE57.758.065.965.1
PMR w/o PER60.361.367.265.6
## PMR61.1    61.867.166.4
Table  2.Comparison  with  various  modulation  strategies  on
CREMA-D and AVE dataset with concatenation and film fusion
methods. PMR achieves the best performance among them.
DatasetCREMA-DCG-MNIST
MethodAccAcc
## MMTM51.671.4
CentralNet50.269.3
## MMTM†55.174.2
CentralNet†
## 52.972.1
Table 3.   Performance on CREMA-D and CG-MNIST with two
kinds of intermediate fusion methods.†indicates PMR strategy is
applied, which achieves better performance.
done in [29].  Compared to them, our method doesn’t need
additional modules and is independent of the fusion meth-
ods  and  classifier  structure,  which  makes  it  applicable  to
more scenarios.
Performance on different architectures.The fusion stage
of the four fusion methods used above is after the encoder
or  the  classifier.   To  validate  the  applicability  of  PMR  in
more scenarios, we combine it with two intermediate fusion
methods MMTM [14] and CentralNet [39] with and without
PMR on CREMA-D and CG-MNIST. MMTM could recali-
brate the channel-wise features of different CNN streams by
squeeze and multimodal excitation steps. We use ResNet18
as the backbone and apply MMTM in the three final residual
blocks.  CentralNet uses both unimodal hidden representa-
tions and a central joint representation at each layer, which
are fused by a learned weighted summation.  We use only
one frame for each video on CREMA-D for convenience.
According  to  Tab.  3,  our  proposed  PMR  achieves  promi-
nent improvement even if fusion is operated during the pro-
cessing of encoders, indicating the applicability of PMR in
complex scenarios.
Application on another task.We process the AVE dataset
as a classification dataset in the above experiments. Here we
use the original AVE dataset to complete the audio-visual
event localization task.  We apply PMR on the AVEL [38]
with concatenation and DMRN [38] fusion methods, which
utilize LSTM to model temporal dependencies in the two
modalities respectively. We evaluate the performance of su-
pervised event localization in the late fusion style.  In ex-
periments, we only apply PMR on the ground truth video
DatasetAVE
FusionAcc
concat72.7
## DMRN73.1
concat†74.3
## DMRN†74.2
Table 4.  Event localization experiments based on the AVEL with
two fusion methods.†indicates PMR strategy is applied.
segments without extra processing on the video segments
which are not relevant to the label. We choose to use Adam
optimizer  with  the  same  settings  in  [38].   The  results  are
shown in Tab. 4.  Applying PMR on the event localization
task  still  achieves  a  certain  promotion  on  accuracy.   The
utilization of two kinds of fusion methods on this task in-
dicates  that  our  method  has  great  application  potential  in
various task scenarios and various fusion methods.
5.4. Ablation study
Uni-modal performance comparison.As we discussed in
Sec. 3, the interaction between modalities eventually leads
to the failure of each modality to effectively exploit its own
features. Therefore, we first compare the uni-modal perfor-
mance among uni-modal trained models and the uni-modal
branch in multimodal models with and without PMR strat-
egy.  As shown in Figs. 4a and 4b, the uni-audio model is
slightly better than the audio branch in vanilla multimodal
learning,  while  the  visual  branch  has  few  effective  learn-
ing resulting in a distinct gap with uni-visual modal.  After
applying PMR strategy, the audio branch doesn’t show a no-
ticeable change since our strategy is mainly for facilitation
for the slow-learning modality, while the visual branch im-
proves considerably, indicating that our PMR strategy out-
performs  by  indeed  promoting  the  feature  exploitation  of
the  slow-learning  modality.   Fig.  4c  illustrates  the  curves
of imbalance ratioρin vanilla multimodal and multimodal
with PMR. It can be seen that our PMR helps to alleviate
the modality imbalance in multimodal learning.  Although
audio modality learns fast in the early training process and
converges  quickly,  visual  modality  can  still  get  improve-
ment. Compare with the baseline without PMR, the imbal-
ance ratio with PMR (green line) decreases gradually even
if the dominant modality has already converged, indicating
the intrinsic encouragement from PCE is less affected by
other modalities.
Analysis of subset data scales for prototype calculation.
In our proposed PMR, the computation of prototypes is the
major additional cost.  We analyze the performance of our
method on different scales of data for prototype calculation.
The results are shown in Tab. 5.  When the scale of subset
is greater than 10%,  PMR tends to be stable in accuracy,
which  means  data  of  this  scale  can  represent  the  approx-
## 20035

## 0255075100125
## Epoch
## 0.40
## 0.45
## 0.50
## 0.55
## Accuracy
uni-audio
audio in multimodal
audio in multimodal with PMR
## (a)
## 0255075100125
## Epoch
## 0.2
## 0.3
## 0.4
## 0.5
## Accuracy
uni-visual
visual in multimodal
visual in multimodal with PMR
## (b)
## 0255075100
## Epoch
## 1
## 2
## 3
## 4
## 5
## 6
audio-visual model
audio-visual model with PMR
## (c)
Figure 4.  Performance of the uni-modal trained models, uni-modal branch in multimodal trained models, and uni-modal branch in multi-
modal with PMR on CREMA-D dataset.  The fusion method in multimodal models is concatenation.  (a) Performance of audio modality.
(b) Performance of visual modality. (c) The change of imbalance ratioρ.
DatasetCREMA-DAVE
ScaleAccAcc
baseline53.265.4
## 1%54.162.1
## 5%
## 58.465.8
## 10%61.167.1
## 50%
## 61.267.7
## 100%
## 61.567.3
Table 5. Experiments on CREMA-D and AVE with different sub-
set scales. n% indicates the proportion of the training dataset. Fu-
sion method here is concatenation
imate  distribution  of  the  overall  data  on  CREMA-D  and
AVE. If the data scale is too small, the performance would
drop a little bit, even worse than the vanilla baseline when
the subset data scale is just 1% on AVE. The reason may be
the calculated prototype is biased because of limited data,
which further hinders the training of model.  Although our
method introduces a certain amount of extra computation,
the required computation is not much and can be reduced
with a reasonable selection of data.
Adaptive optimizers.We analyze the modality imbalance
problem and propose the PMR strategy under the assump-
tion:  the same learning rate for each gradient calculation,
which  is  true  for  popular  SGD-based  algorithms,  but  not
rigorous for adaptive optimization algorithms.  To validate
the effectiveness of our method on more adaptive methods,
we apply PMR on optimizers AdaGrad [6] and Adam [17],
dynamically adjusting the learning rate for each parameter.
As demonstrated in Tab. 6, we empirically show that PMR
works with different optimizers.  Different optimizers per-
form inconsistently on different datasets.
## 6. Discussion
Multimodal learning usually falls into a suboptimal so-
lution because of the modality imbalance problem, which
indicates that vanilla optimization with a single strategy for
DatasetCREMA-DAVE
OptimizerAccAcc
## SGD53.265.4
AdaGrad
## 58.865.1
## Adam59.864.4
## SGD†61.167.1
AdaGrad†
## 61.566.2
## Adam†
## 65.366.5
Table  6.   Experiments  with  AdaGrad  and  Adam  optimizers  on
CREMA-D and AVE.†indicates PMR strategy is applied.  PMR
consistently achieves better performance.
different modalities is limited.  We propose the prototypi-
cal modal rebalance (PMR) strategy to introduce different
learning strategies for different modalities, i.e., accelerating
the  slow  modality  with  prototypical  cross  entropy  (PCE)
loss  and  reducing  the  inhibition  from  dominant  modality
with prototypical entropy regularization (PER) term.  This
method  achieves  considerable  performance  improvement
on the three multimodal datasets with different model struc-
tures  and  fusion  methods.   The  non-parametric  classifiers
with prototypes can be applied in any scenario as long as
we have the representations of instances for each modality.
## 7. Acknowledgement
Our  work  was  supported  by  fundings  from  the  Key-
Area  Research  and  Development  Program  of  Guangdong
Province (No.  2021B0101400003), Hong Kong RGC Re-
search Impact Fund (No.  R5060-19), Areas of Excellence
Scheme  (AoE/E-601/22-R),  General  Research  Fund  (No.
152203/20E,  152244/21E,  152169/22E,  PolyU15222621),
Shenzhen Science and Technology Innovation Commission
(JCYJ20200109142008673) and the grant from Establish-
ment  of  Distributed  Artificial  Intelligence  Laboratory  for
Interdisciplinary Research (UGC/IDS(R)11/19).
## 20036

## References
## [1]  Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak,
Orestis Plevrakis, and Nikunj Saunshi. A theoretical analysis
of contrastive  unsupervised representation learning.arXiv
preprint arXiv:1902.09229, 2019. 2
## [2]  Houwei  Cao,   David  G  Cooper,   Michael  K  Keutmann,
Ruben C Gur, Ani Nenkova, and Ragini Verma.   Crema-d:
Crowd-sourced emotional multimodal actors dataset.IEEE
transactions on affective computing, 5(4):377–390, 2014.  3,
## 5
[3]  Yuhan Chai, Lei Du, Jing Qiu, Lihua Yin, and Zhihong Tian.
Dynamic prototype network based on sample adaptation for
few-shot malware detection.IEEE Transactions on Knowl-
edge and Data Engineering, 2022. 2
## [4]  Kaize Ding, Jianling Wang, Jundong Li, Kai Shu, Chenghao
Liu, and Huan Liu. Graph prototypical networks for few-shot
learning on attributed networks.  InProceedings of the 29th
ACM International Conference on Information & Knowledge
Management, pages 295–304, 2020. 2
## [5]  Chenzhuang Du, Tingle Li, Yichen Liu, Zixin Wen, Tianyu
Hua,   Yue   Wang,   and   Hang   Zhao.Improving   multi-
modal  learning  with  uni-modal  teachers.arXiv  preprint
arXiv:2106.11059, 2021. 2
[6]  John Duchi, Elad Hazan, and Yoram Singer.  Adaptive sub-
gradient  methods  for  online  learning  and  stochastic  opti-
mization.Journal of machine learning research, 12(7), 2011.
## 8
[7]  Tianyu Gao, Xu Han, Zhiyuan Liu, and Maosong Sun.  Hy-
brid  attention-based  prototypical  networks  for  noisy  few-
shot relation classification. InProceedings of the AAAI Con-
ference on Artificial Intelligence, 2019. 2
[8]  Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Ba-
tra, and Devi Parikh.  Making the v in vqa matter: Elevating
the role of image understanding in visual question answer-
ing.   InProceedings  of  the  IEEE  conference  on  computer
vision and pattern recognition, pages 6904–6913, 2017. 2
[9]  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition.   InProceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 6
[10]  Javed Imran and Balasubramanian Raman. Evaluating fusion
of rgb-d and inertial sensors for multimodal human action
recognition.Journal of Ambient Intelligence and Humanized
## Computing, 11(1):189–208, 2020. 1
[11]  Giridharan  Iyengar  and  Harriet  J  Nock.Discriminative
model fusion for semantic concept detection and annotation
in video.  InProceedings of the eleventh ACM international
conference on Multimedia, pages 255–258, 2003. 6
[12]  Allan  Jabri,  Armand  Joulin,  and  Laurens  van  der  Maaten.
Revisiting visual question answering baselines. InEuropean
conference  on  computer  vision,  pages  727–739.  Springer,
## 2016. 2
[13]  Yu-Gang  Jiang,  Zuxuan  Wu,  Jinhui  Tang,  Zechao  Li,  Xi-
angyang  Xue,  and  Shih-Fu  Chang.   Modeling  multimodal
clues in a hybrid deep learning framework for video classi-
fication.IEEE Transactions on Multimedia,  20(11):3137–
## 3147, 2018. 1
## [14]  Hamid Reza Vaezi Joze, Amirreza Shaban, Michael L Iuz-
zolino, and Kazuhito Koishida. Mmtm: Multimodal transfer
module  for  cnn  fusion.   InProceedings  of  the  IEEE/CVF
Conference  on  Computer  Vision  and  Pattern  Recognition,
pages 13289–13299, 2020. 7
[15]  Douwe  Kiela,  Edouard  Grave,  Armand  Joulin,  and  Tomas
Mikolov.Efficient  large-scale  multi-modal  classification.
InProceedings of the AAAI Conference on Artificial Intel-
ligence, 2018. 6
## [16]  Byungju Kim, Hyunwoo Kim, Kyungsu Kim, Sungjin Kim,
and Junmo Kim. Learning not to learn: Training deep neural
networks with biased data. InProceedings of the IEEE/CVF
Conference  on  Computer  Vision  and  Pattern  Recognition,
pages 9012–9020, 2019. 6
[17]  Diederik P Kingma and Jimmy Ba.   Adam:  A method for
stochastic  optimization.arXiv  preprint  arXiv:1412.6980,
## 2014. 8
[18]  Yann  LeCun,  L
## ́
eon  Bottou,  Yoshua  Bengio,  and  Patrick
Haffner. Gradient-based learning applied to document recog-
nition.Proceedings of the IEEE, 86(11):2278–2324, 1998.
## 6
## [19]  Junnan   Li,   Ramprasaath   Selvaraju,   Akhilesh   Gotmare,
Shafiq  Joty,  Caiming  Xiong,  and  Steven  Chu  Hong  Hoi.
Align before fuse: Vision and language representation learn-
ing with momentum distillation.Advances in neural infor-
mation processing systems, 34:9694–9705, 2021. 3
[20]  Junnan Li, Pan Zhou, Caiming Xiong, and Steven CH Hoi.
Prototypical contrastive learning of unsupervised representa-
tions.arXiv preprint arXiv:2005.04966, 2020. 2
[21]  Dong Liu, Kuan-Ting Lai, Guangnan Ye, Ming-Syan Chen,
and Shih-Fu Chang.   Sample-specific late fusion for visual
category recognition.   InProceedings of the IEEE Confer-
ence  on  Computer  Vision  and  Pattern  Recognition,  pages
## 803–810, 2013. 6
## [22]  Xinwang  Liu,  Xinzhong  Zhu,  Miaomiao  Li,  Lei  Wang,
## Chang Tang, Jianping Yin, Dinggang Shen, Huaimin Wang,
and Wen Gao. Late fusion incomplete multi-view clustering.
IEEE transactions on pattern analysis and machine intelli-
gence, 41(10):2410–2423, 2018. 3
[23]  Youssef  Mroueh,  Etienne  Marcheret,  and  Vaibhava  Goel.
Deep multimodal learning for audio-visual speech recogni-
tion.  In2015 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages 2130–2134.
## IEEE, 2015. 1
## [24]  Jiquan  Ngiam,  Aditya  Khosla,  Mingyu  Kim,  Juhan  Nam,
Honglak Lee,  and Andrew Y Ng.   Multimodal deep learn-
ing. InICML, 2011. 1
## [25]  Dan Oneat ̧
## ̆
a and Horia Cucu.  Improving multimodal speech
recognition  by  data  augmentation  and  speech  representa-
tions.   In2022 IEEE/CVF Conference on Computer Vision
and Pattern Recognition Workshops (CVPRW), pages 4578–
## 4587. IEEE, 2022. 1
[26]  Andrew  Owens  and  Alexei  A  Efros.    Audio-visual  scene
analysis with self-supervised multisensory features.  InPro-
ceedings  of  the  European  Conference  on  Computer  Vision
(ECCV), pages 631–648, 2018. 6
[27]  Yingwei  Pan,  Ting  Yao,  Yehao  Li,  Yu  Wang,  Chong-Wah
Ngo,  and  Tao  Mei.Transferrable  prototypical  networks
## 20037

for  unsupervised  domain  adaptation.InProceedings  of
the  IEEE/CVF  conference  on  computer  vision  and  pattern
recognition, pages 2239–2247, 2019. 2
[28]  Yagya Raj Pandeya, Bhuwan Bhattarai, and Joonwhoan Lee.
Deep-learning-based  multimodal  emotion  classification  for
music videos.Sensors, 21(14):4927, 2021. 1
[29]  Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and
Di Hu. Balanced multimodal learning via on-the-fly gradient
modulation.   InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8238–
## 8247, 2022. 1, 2, 3, 4, 6, 7
## [30]  Ethan  Perez,  Florian  Strub,  Harm  De  Vries,  Vincent  Du-
moulin, and Aaron Courville.  Film: Visual reasoning with a
general conditioning layer. InProceedings of the AAAI Con-
ference on Artificial Intelligence, 2018. 6
[31]  Herbert Robbins and Sutton Monro.   A stochastic approxi-
mation method.The annals of mathematical statistics, pages
## 400–407, 1951. 6
[32]  Vivek  Roy,  Yan  Xu,  Yu-Xiong  Wang,  Kris  Kitani,  Rus-
lan  Salakhutdinov,  and  Martial  Hebert.Few-shot  learn-
ing  with  intra-class  knowledge  transfer.arXiv  preprint
arXiv:2008.09892, 2020. 2
[33]  Amir Shahroudy, Tian-Tsong Ng, Yihong Gong, and Gang
Wang.  Deep multimodal feature analysis for action recogni-
tion in rgb+ d videos.IEEE transactions on pattern analysis
and machine intelligence, 40(5):1045–1058, 2017. 1
## [34]  Shashi Kant Shankar, Luis P Prieto, Mar
## ́
ıa Jes
## ́
us Rodr
## ́
ıguez-
Triana,  and Adolfo Ruiz-Calleja.   A review of multimodal
learning  analytics  architectures.In2018  IEEE  18th  in-
ternational  conference  on  advanced  learning  technologies
(ICALT), pages 212–214. IEEE, 2018. 1
[35]  Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical
networks for few-shot learning.Advances in neural informa-
tion processing systems, 30, 2017. 2, 4
[36]  Nitish  Srivastava  and  Russ  R  Salakhutdinov.   Multimodal
learning with deep boltzmann machines.Advances in neural
information processing systems, 25, 2012. 1
[37]  Haiman  Tian,  Yudong  Tao,  Samira  Pouyanfar,  Shu-Ching
Chen,  and  Mei-Ling  Shyu.    Multimodal  deep  representa-
tion  learning  for  video  classification.World  Wide  Web,
## 22(3):1325–1341, 2019. 1
[38]  Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-
liang Xu.  Audio-visual event localization in unconstrained
videos. InProceedings of the European Conference on Com-
puter Vision (ECCV), pages 247–263, 2018. 1, 6, 7
## [39]  Valentin  Vielzeuf,  Alexis  Lechervy,  St
## ́
ephane  Pateux,  and
## Fr
## ́
ed
## ́
eric Jurie.   Centralnet:  a multilayer approach for mul-
timodal fusion.  InProceedings of the European Conference
on Computer Vision (ECCV) Workshops, pages 0–0, 2018. 7
[40]  Weiyao Wang, Du Tran, and Matt Feiszli. What makes train-
ing multi-modal classification networks hard?   InProceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 12695–12705, 2020. 1, 2, 6
[41]  Thomas  Winterbottom,  Sarah  Xiao,  Alistair  McLean,  and
Noura Al Moubayed.  On modality bias in the tvqa dataset.
arXiv preprint arXiv:2012.10210, 2020. 2
[42]  Nan   Wu,   Stanislaw   Jastrzebski,   Kyunghyun   Cho,   and
Krzysztof  J  Geras.Characterizing  and  overcoming  the
greedy  nature  of  learning  in  multi-modal  deep  neural  net-
works.   InInternational Conference on Machine Learning,
pages 24043–24055. PMLR, 2022. 1, 2, 6
[43]  Yan Xia and Zhou Zhao.  Cross-modal background suppres-
sion for audio-visual event localization.   InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 19989–19998, 2022. 1
## [44]  Fanyi Xiao, Yong Jae Lee, Kristen Grauman, Jitendra Malik,
and Christoph Feichtenhofer. Audiovisual slowfast networks
for  video  recognition.arXiv  preprint  arXiv:2001.08740,
## 2020. 1, 2, 6
[45]  Yuzhe Yang, Hao Wang, and Dina Katabi. On multi-domain
long-tailed  recognition,  generalization  and  beyond.arXiv
preprint arXiv:2203.09513, 2022. 2
[46]  Litian  Zhang,  Xiaoming  Zhang,  and  Junshu  Pan.   Hierar-
chical  cross-modality  semantic  correlation  learning  model
for multimodal summarization.  InProceedings of the AAAI
Conference on Artificial Intelligence, 2022. 3
## 20038