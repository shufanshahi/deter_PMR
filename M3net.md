

Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural
Networks for Emotion Recognition in Conversation
## Feiyu Chen
## †‡
## Jie Shao
## †‡∗
## Shuyuan Zhu
## †
## Heng Tao Shen
## †‡
## †
University of Electronic Science and Technology of China, Chengdu, China
## ‡
## Sichuan Artificial Intelligence Research Institute, Yibin, China
## {chenfeiyu,shaojie,eezsy,shenhengtao}@uestc.edu.cn
## Abstract
Complex relationships of high arity across modality and
context  dimensions  is  a  critical  challenge  in  the  Emotion
Recognition  in  Conversation  (ERC)  task.Yet,  previous
works tend to encode multimodal and contextual relation-
ships  in  a  loosely-coupled  manner,  which  may  harm  re-
lationship  modelling.    Recently,  Graph  Neural  Networks
(GNN) which show advantages in capturing data relations,
offer a new solution for ERC. However, existing GNN-based
ERC models fail to address some general limits of GNNs,
including assuming pairwise formulation and erasing high-
frequency signals, which may be trivial for many applica-
tions but crucial for the ERC task.  In this paper, we pro-
pose a GNN-based model that explores multivariate rela-
tionships and captures the varying importance of emotion
discrepancy  and  commonality  by  valuing  multi-frequency
signals.  We empower GNNs to better capture the inherent
relationships among utterances and deliver more sufficient
multimodal and contextual modelling. Experimental results
show that our proposed method outperforms previous state-
of-the-art works on two popular multimodal ERC datasets.
## 1. Introduction
Human  beings  constantly  express  their  feelings  in  ev-
eryday  communication.   Emotion  Recognition  in  Conver-
sation (ERC) aims at enabling machines to detect interac-
tive human emotions in a dialogue, utilizing multi-sensory
data,  including  textual,  visual  and  acoustic  information
[5, 13, 18, 24].  Unlike traditional affective computing tasks
that are performed on single modalities (e.g., text, speech
or facial images) [12, 28, 32] or/and in non-conversational
## ∗
Corresponding  author:   Jie  Shao.    This  work  is  supported  by  the
National  Natural  Science  Foundation  of  China  (No.61832001  and
No.   62276047),  Natural  Science  Foundation  of  Sichuan  Province  (No.
2023NSFSC1972) and Science and Technology Program of Yibin Sanjiang
New Area (No. 2023SJXQYBKJJH001).
## 
## 3
## 澳
## 
## 1
## 澳
Oh what, you-you want
both of them?
## Rachel Karen Green,
where's the other earring?!
Okay, okay, okay, look,
just don't freak out, but I
kinda lost it.
I know it's in the
apartment, but I definitely
lost it.
Well, what am I going to
tell Monica? She wants to
wear them tonight!
## Anger
## Fear
## Neutral
## Fear
## 
## 4
## 澳
## 
## 2
## 澳
## 
## 5
## 澳
## Dialogue
## Emotion
## Surprise
Low voice
Soft voice
Loud voive
Flat tone
High pitch
## TAV
prediction
prediction
Visual informationTextual informationAcoustic information
Wait, Rach! Where's the
other one?
Calm voice
## 
## 6
## 澳
## Neutral
Figure 1. An example of multimodal dialogue (left) and the com-
plex multivariate relationships ofu
## 3
andu
## 6
## (right).
scenario  [15, 23, 33],  there  exists  a  distinct  and  essential
challenge  in  the  ERC  task  -  the  complex  multivariate  re-
lationships  among  multiple  modalities  and  conversational
context.  In other words, the emotional dependencies of an
utterance are usually of high arity, and involve multi-source
information across both modality and context dimensions.
Figure  1  presents  a  sample  of  conversation  between  two
speakers.  Take the utteranceu
## 3
as an example.  The visual
and  acoustic  messages  of  utteranceu
## 3
(an  expressionless
face and a flat tone) are ambiguous, but imply a veiled anger
if coupled with the text.  Moreover, the emotion behindu
## 3
is also related to the preceding contextu
## 1
andu
## 2
. In partic-
ular, the change from calling by nickname inu
## 1
to calling
by full name inu
## 3
suggests an emotion shift caused byu
## 2
## ,
since another speaker tries to make a joke with a pretended
lightness.   Therefore,  the relationships in{u
## 1
## ,u
## 2
## ,u
## 3
## }are
complex  and  multivariate,  and  involve  interdependencies
across both modality and context dimensions.
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
## 10761

Researchers  have  been  exploring  how  to  capture  the
complex  relationships  more  effectively.   Among  existing
ERC models, a dominant paradigm is to capture contextual
relationships with context-sensitive modules such as recur-
rent  unit  or  transformer,  whilst  modelling  multimodal  re-
lationships through various fusion methods [4, 24, 25, 34].
Despite the advances, this paradigm tends to underrate mul-
tivariate relationships among modalities and context,  as it
limits the natural interaction between loosely-coupled mul-
timodal and contextual modelling.
More  recently,  Graph  Neural  Networks  (GNNs)  have
shown great promise and yielded remarkable improvements
in ERC, by revealing rich expressive power of mining struc-
tural information and data relations [17, 18].  A routine so-
lution  is  to  construct  a  heterogeneous  graph  where  each
modality  of  an  utterance  is  regarded  as  a  node,  and  con-
nected with other modalities of the same utterance as well as
connected with the utterances in same modality in the same
dialogue. Carefully-tweaked edge-weighting strategies usu-
ally follow.  On this basis, multimodal and contextual de-
pendencies among utterances can be modelled simultane-
ously through message passing, and thus deliver tighter en-
tanglement and richer interaction. Powerful as these GNN-
based methods are, they still suffer from two limitations:
i)Insufficient  multivariate  relationships.Conven-
tional  GNNs  assume  pairwise  relationships  of  ob-
jects of interest, and can only offer an approximation
of higher-order and multivariate relationships through
multiple pairs [1, 10].  However, degeneration of those
multivariate  relationships  into  pairwise  formulation
may harm the expressiveness [20, 30]. Therefore, com-
plex multivariate relationships in ERC may not be suf-
ficiently modelled by previous GNN-based methods.
ii)Underestimated high-frequency information.  It has
been shown that the propagation rule of GNNs (i.e., ag-
gregating and smoothing messages from neighbours) is
an analogy to a fixed low-pass filter [26, 31], and it is
mainly low-frequency messages that flow in the graph
while  the  effects  of  high-frequency  ones  are  much
weakened.   Moreover,  Boet  al.  [2]  show  that  low-
frequency messages, which retain the commonality of
node features, perform better on assortative graphs (in
which  the  linked  nodes  tend  to  have  similar  features
and share the same label).  In contrast, high-frequency
information that mirrors discrepancy and inconsistency
is  more  crucial  on  disassortative  graphs.   For  ERC,
the constructed graphs are in general highly disassorta-
tive, where inconsistent emotional messages may exist
among  modalities  (say  being  sarcastic)  or  short-term
context.  Hence, high-frequency information may pro-
vide crucial guidance, which is however badly ignored
by previous GNN-based ERC models, incurring bottle-
neck of performance improvement.
To   address   these   issues,   in   this   work   we   propose
MultivariateMulti-frequencyMultimodal  Graph  Neural
Network  (M
## 3
Net),  which  aims  to  capture  more  sufficient
multivariate  relationships  among  modalities  and  context,
while benefiting from multi-frequency information within
the graph.  At the core of M
## 3
Net are two parallel compo-
nents, multivariate propagation and multi-frequency propa-
gation.  Concretely, we first construct a hypergraph neural
network with edge-dependent node weights [7] for multi-
variate propagation, in which each modality of an utterance
is represented as a node. We construct multimodal and con-
textual hyperedges, which can connect arbitrary number of
nodes, and thus can naturally encode relationships of higher
arity.   Meanwhile,  we model multi-frequency information
upon an undirected GNN, by adapting a set of frequency fil-
ters [2, 8] to distil different frequency constituents from the
node features.  We adaptively integrate different frequency
signals to capture the varying importance of emotion dis-
crepancy and emotion commonality in the local neighbour-
hood, so as to achieve adaptive information sharing pattern.
The effectiveness of our work is further demonstrated by
extensive experimental studies on two popular multimodal
ERC  datasets  IEMOCAP  [3]  and  MELD  [27].   We  show
that M
## 3
Net outperforms previous state-of-the-art methods.
- Related work
2.1. Emotion recognition in conversation
Due  to  the  great  potential  in  interactive  applications,
Emotion Recognition in Conversation (ERC) has attracted
great interests of many researchers.  Various attempts have
been made to study multimodal and contextual relationships
in ERC. Some early works [13,14,24] focused more on con-
textual dependencies and conducted simple feature concate-
nation to perform multimodal modelling. To enhance the in-
terrelation between modalities and context, recent methods
introduced more advanced schemes such as positional at-
tention [34] and adaptive computation [5].  However, these
methods  still  encode  multimodal  and  contextual  relation-
ships  in  a  loosely-coupled  manner,  which  may  result  in
weak interaction between them.   More recently,  some re-
searchers formulated the ERC task upon GNNs, which are
powerful  in  mining  data  relations  hence  exhibit  superior
capability to model contextual and multimodal dependen-
cies [17, 18].  Nevertheless, these GNN-based models still
deliver insufficient multivariate relationships and underrate
high-frequency signals, as we discussed.
In this work, we propose a new approach that enhances
multivariate  information  among  modalities  and  context,
whilst capturing the varying importance of emotion discrep-
ancy and emotion commonality, to deliver more sufficient
multimodal and contextual modelling.
## 10762

## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## 33
## 
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## 33
## 
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳ℎ
## 1
## 
## 澳ℎ
## 2
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 1
## 
## ℎ
## 2
## 
## ℎ
## 3
## 
## ℎ
## 1
## 
## ℎ
## 2
## 
## ℎ
## 3
## 
## ℎ
## 1
## 
## ℎ
## 2
## 
## ℎ
## 3
## 
## ℎℎ
## 11
## 
## ℎℎ
## 22
## 
## ℎℎ
## 33
## 
## ℎℎ
## 11
## 
## ℎℎ
## 22
## 
## ℎℎ
## 33
## 
## ℎℎ
## 11
## 
## ℎℎ
## 22
## 
## ℎℎ
## 33
## 
## 
## 1
## 
## 澳
## 2
## 
## 澳
## 3
## 
## 澳
## 
## 3
## 
## 澳
## 2
## 
## 澳
## 1
## 
## 澳
## 1
## 
## 澳澳
## 2
## 
## 澳
## 3
## 
## 澳
## 
## 1
## 
## 澳
## 2
## 
## 澳
## 3
## 
## 澳
## 3
## 
## 
## 2
## 
## 
## 1
## 
## 
## 1
## 
## 澳
## 2
## 
## 澳
## 3
## 
low
high
## 
## 3
## 
## 
## 1
## 
## 澳
## 
## 2
## 
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
e
## 1
e
## 2
e
## 3
e
n
## Ă
## Ă
## 
## 1
## 澳
## 
## 4
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
## 澳澳
## 1
## 
## 澳
## 
## 2
## 
## 澳
## 3
## 
## 澳
## 
## 3
## 
## 澳
## 
## 2
## 
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
## 澳澳
e
## 1
e
## 2
e
## 3
e
n
## Ă
## Ă
## 
## 1
## 澳
## 
## 4
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
## 澳澳
## 1
## 
## 澳
## 
## 2
## 
## 澳
## 3
## 
## 澳
## 
## 3
## 
## 澳
## 
## 2
## 
## 澳
## 
## 1
## 
## 澳
## 
## 1
## 
## 澳澳
## ̅

## 
## 澳
## ̅

## 
## ̅

## 
## 澳
## 
## ̅

## 
## 澳
## 
## ̅

## 
## 澳
## 
## ̅

## 
## Classifier
## Multivariate Propagation
## Multi-frequency Propagation
## Embedding
## Speaker
## Embedding



## 澳
Modality EncodingEmotion Classifier
## Hypergraph Conv

## 1
## 
## 澳

## 2
## 

## 3
## 

## 3
## 
## 澳

## 2
## 
## 澳

## 1
## 

## 1
## 

## 2
## 
## 澳

## 3
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 2
## 
## 澳
## ℎ
## 1
## 
## 澳
## ℎ
## 1
## 
## 澳
## ℎ
## 2
## 
## 澳
## ℎ
## 3
## 
## 澳
## ℎ
## 3
## 
## 澳
## ℎ
## 2
## 
## 澳
## ℎ
## 1
## 
## 澳
## FC
## GRU
## GRU
## GRU
## FC

## 1
## 
## 澳

## 2
## 

## 3
## 

## 3
## 
## 澳

## 2
## 
## 澳

## 1
## 

## 1
## 

## 2
## 
## 澳

## 3
## 
## 澳ℎ
## 3
## 
## 澳
## ℎ
## 2
## 
## 澳
## ℎ
## 1
## 
## 澳
## ℎ
## 1
## 
## 澳
## ℎ
## 2
## 
## 澳
## ℎ
## 3
## 
## 澳
## ℎ
## 3
## 
## 澳
## ℎ
## 2
## 
## 澳
## ℎ
## 1
## 
## 澳
## FC
## GRU
## GRU
## GRU
## FC
## Notations:
Visual modalityTextual modalityAcoustic modality
## Concatenation
## Notations:
Visual modalityTextual modalityAcoustic modality
## Concatenation
Figure 2. Detailed architecture of the proposed M
## 3
## Net.
2.2. Graph neural networks
Graph Neural Networks (GNNs) have a distinct advan-
tage in modelling data relationships, and have been widely
employed in various applications such as recommendation
[16] and action recognition [6].  GNNs have also inspired
ERC  researchers  and  offer  a  new  solution  for  the  ERC
task, from unimodal setting [12, 28] to multimodal scenario
[17, 18].  However, previous works fail to address the gen-
eral limits of GNNs, including conducting pairwise formu-
lation and erasing high-frequency information, which mo-
tivates our work.  We present a GNN-based model that en-
codes relationships of higher arity and values different fre-
quency signals in the neighbourhood.  We empower GNNs
to better capture the inherent relations among utterances and
boost up the performance.
## 3. Methodology
In   a   nutshell,   an   ERC   model   aims   to   detect   the
emotion  state  of  each  utterance  in  a  dialogue.For-
mally,  a  dialogue  contains  a  sequence  ofNutterances
## {(u
## 1
## ,p
## 1
## ),(u
## 2
## ,p
## 2
## ),...,(u
## N
## ,p
## N
)}, where each utteranceu
i
## ,
spoken  by  speakerp
i
,  consists  of  multi-sensory  data,  in-
cluding textual (u
t
i
), visual (u
v
i
) and acoustic (u
a
i
) modal-
ities.  The goal is to predict the emotion category of each
constituent utteranceu
i
from a predefined set ofCclasses.
Figure 2 shows the architecture of the proposed M
## 3
## Net.
In general, M
## 3
Net contains four components: modality en-
coding, multivariate propagation, multi-frequency propaga-
tion, and an emotion classifier.
3.1. Modality encoding
A  conversation  is  sequential  in  nature  and  consists  of
multiple speakers.  Therefore, we firstly process unimodal
utterances  with  speaker  and  context  information,  to  ob-
tain speaker- and context-aware unimodal representations.
Specifically, we denote each speaker with a one-hot vector
s
i
and maintain a lookup table forMspeakers to calculate
the speaker embeddingS
i
at thei-th conversation turn:
## S
i
## =W
s
s
i
## ,(1)
in whichS
i
## ∈R
## D
h
andW
s
is trainable weight. In addition,
we  employ  a  bidirectional  Gated  Recurrent  Unit  (GRU)
to encode the conversational textual features.  We empiri-
cally observe that encoding visual and acoustic modalities
with recurrent modules has no positive effect on the perfor-
mance, hence use two one-hidden-layer multilayer percep-
tronsW
## 1
andW
## 2
to encode acoustic and visual modalities
respectively. Mathematically,
c
t
i
## =
## ←−−→
GRU(u
t
i
## ,c
t
i(+,−)1
## ),
c
a
i
## =W
## 1
u
a
i
## +b
a
i
## ,
c
v
i
## =W
## 2
u
v
i
## +b
v
i
## ,
## (2)
in whichc
t
i
## ,c
a
i
## ,c
v
i
## ∈R
## D
h
. We then add speaker embedding
to obtain speaker- and context-aware unimodal representa-
tions{h
t
i
## ,h
a
i
## ,h
v
i
}at thei-th conversation turn:
h
x
i
## =c
x
i
## +S
i
,  x∈{t,a,v}.(3)
## 10763

3.2. Multivariate propagation
The  main  idea  of  the  multivariate  propagation  module
is  to  explore  the  multivariate  and  high-order  information
among multiple modalities and conversational context.  We
begin by constructing a hypergraphHwith edge-dependent
node weights, from the sequentially encoded utterances.
3.2.1    Graph construction
Generally, given a sequence of utterances withNconversa-
tion turns, we construct a hypergraphH= (V
## H
## ,E
## H
## ,ω,γ),
in which each nodev∈V
## H
## (|V
## H
|= 3N)corresponds to a
unimodal utterance, and every hyperedgee∈ E
## H
## (|E
## H
## |=
3 +N)encodes multimodal or contextual dependencies. A
weightω(e)is assigned for every hyperedgee∈E
## H
, and a
weightγ
e
(v)for every hyperedgee∈E
## H
and every nodev
incident toe.  LetH∈R
## |V
## H
## |×|E
## H
## |
represent the incidence
matrix, in which a nonzero entryH
ve
= 1indicates that the
hyperedgeeis incident with the nodev; otherwiseH
ve
## = 0.
Nodes:Each modality of an utterance is represented as a
node in hypergraph, i.e.,v
t
i
for the textual modality,v
a
i
for
the acoustic modality andv
v
i
for the visual modality.  We
initialize the node embeddings{v
t
i
## ,v
a
i
## ,v
v
i
}with the sequen-
tially encoded representations{h
t
i
## ,h
a
i
## ,h
v
i
## }respectively.
Hyperedges:The design of hyperedges is based on the
assumption  that  the  emotion  behind  an  utterance  in  a  di-
alogue  is  mainly  determined  by  the  joint  effect  of  multi-
ple modalities and conversational context, and multivariate
relationships  may  exist  across  both  dimensions.   To  fully
investigate the complex multivariate relationships, we con-
struct  multimodal  hyperedges  and  contextual  hyperedges
for each node. Concretely, as shown in Figure 2, each node
v
x
i
(x∈ {t,a,v})is  firstly  connected  to  all  other  utter-
ances in the same modality in the same dialogue{v
x
j
## |j∈
[1,N],j6=i}, with one contextual hyperedge.  Moreover,
each nodev
x
i
is connected to other modalities of the same
utterances{v
z
i
|z∈ {t,a,v},z6=x}, with one multimodal
hyperedge.   In this fashion,  the constructed hypergraph is
able to capture high-order and multivariate messages that
are beyond pairwise formulation.
Weights:Unlike  previous  GNN-based  ERC  models
[12, 18] which manually tweak the edge weighting strate-
gies with complicated relation learning or similarity met-
rics,  we  use  randomly  initialized  weight  values  to  avoid
complicating our model.  Specifically, we define two types
of weights in the hypergraph: i) an edge weightω(e)for ev-
ery hyperedgee, and ii) a node weightγ
e
(v)for every hy-
peredgeeincident tov, a.k.a., edge-dependent node weight
## [7].   Intuitively,γ
e
(v)measures  the  contribution  of  node
vto  hyperedgee,  and  thus  reinforces  fine-grained  multi-
modal and contextual dependencies.  Edge-dependent node
weights  can  thus  be  represented  by  a  weighted  incidence
matrix
## ˆ
## H∈R
## |V
## H
## |×|E
## H
## |
## :
## ˆ
## H=
## {
γ
e
(v),if hyperedgeeis incident with nodev;
## 0,otherwise.
## (4)
3.2.2    Neighbour aggregation
We  reformulate  hypergraph  convolution  operation  [1]  to
propagate multivariate embeddings. We also remove feature
transformation at each iteration as it is observed to be of lit-
tle benefit.  Specifically, we first conduct node convolution
by aggregating node features to update hyperedge embed-
dings,  and  then  conduct  hyperedge  convolution  to  spread
the hyperedge messages to the nodes. Mathematically,
## V
## (l+1)
=σ(D
## −1
## H
## HW
e
## B
## −1
## ˆ
## H
## >
## V
## (l)
## ),(5)
in  whichV
## (l)
## ={v
x
i,(l)
|i∈[1,N],x∈ {t,a,v}} ∈
## R
## |V
## H
## |×D
h
is the input at layerl.σis a non-linear activation
function.W
e
## =diag(w(e
## 1
## ),...,w(e
## |E
## H
## |
))is the hyperedge
weight matrix.D
## H
## ∈R
## |V
## H
## |×|V
## H
## |
andB∈R
## |E
## H
## |×|E
## H
## |
are
the  node  degree  matrix  and  hyperedge  degree  matrix,  re-
spectively.  By this means, the high-order multimodal and
contextual relationships are gradually refined. AfterLiter-
ations, we get the outputs of the last iterationv
x
i,(L)
as the
multivariate representations:
v
t
i
## =v
t
i,(L)
## ,
v
a
i
## =v
a
i,(L)
## ,
v
v
i
## =v
v
i,(L)
## .(6)
3.3. Multi-frequency propagation
The  above  multivariate  propagation  module  is  able  to
capture high-order dependencies that are beyond pairwise,
but it still follows the generic graph learning protocol which
aggregates and smooths signals from the local neighbour-
hood.   This  can  be  interpreted  as  a  form  of  low-pass  fil-
ter and the smoothness of messages is basically spreading
low-frequency  information  whilst  erasing  high-frequency
information [2, 26, 31]. However, as discussed earlier, high-
frequency information that mirrors emotion discrepancy of
nodes may be crucial for ERC, and combining the power
of messages with varying frequencies is worth exploring. It
thus motivates us to propose a multi-frequency propagation
module to distil different frequency constituents with vary-
ing importance.  For this purpose, we further construct an
undirected graphG= (V
## G
## ,E
## G
)from the sequentially en-
coded utterances, in parallel with the multivariate module.
3.3.1    Graph construction
We  construct  an  undirected  graphG= (V
## G
## ,E
## G
## )whose
nodesV
## G
are  identical  to  the  ones  inH,  denoted  with
## {f
t
i
## ,f
a
i
## ,f
v
i
}.   The  node  embeddings  at  the  first  iteration
are initialized with the sequentially encoded representations
## {h
t
i
## ,h
a
i
## ,h
v
i
}as  well.    Different  fromH,  we  construct  a
## 10764

set of edgesE
## G
with pairwise connections.  Similarly,  we
connect  each  nodef
x
i
to  all  other  utterances  in  the  same
modality in the same dialogue{f
x
j
|j∈[1,N],j6=i}, as
well as to other modalities of the same utterances{f
z
i
## |z∈
{t,a,v},z6=x}.   The  constructed  graphGis  shown  in
Figure  2,  with  adjacency  matrixA∈R
## |V
## G
## |×|V
## G
## |
## .   The
normalized graph Laplacian matrix can be represented as
## L=I−D
## −1/2
## G
## AD
## −1/2
## G
,  whereD
## G
## ∈R
## |V
## G
## |×|V
## G
## |
is  a
diagonal degree matrix andIis an identity matrix.
3.3.2    Multi-frequency filtering
We first design a low-pass filterF
l
and a high-pass filterF
h
to distil the signals from the node features:
## F
l
## =I+D
## −1/2
## G
## AD
## −1/2
## G
## = 2I−L,
## F
h
## =I−D
## −1/2
## G
## AD
## −1/2
## G
## =L.
## (7)
It  can  be  noticed  that  the  high-pass  filter  is  equivalent  to
the normalized graph Laplacian matrix, which is consistent
with the theory in image signal processing that the Lapla-
cian  kernel  can  be  employed  to  highlight  high-frequency
edge  information.   According  to  theory  of  graph  Fourier
transform [2, 29],  given a signalφ,  the filtering operation
byF
l
andF
h
can be regarded as the convolutional∗
## C
be-
tweenφand corresponding convolutional kernels:
## F
l
## ∗
## C
φ=F
l
·φ,F
h
## ∗
## C
φ=F
h
## ·φ.(8)
3.3.3    Graph learning
After obtaining low-pass and high-pass filters, we leverage
the  filters  to  adaptively  aggregate  messages  with  varying
frequencies.  Specifically, we use a weighted sum to com-
bine low-frequency and high-frequency messages:
## F
## (k+1)
## =R
l
## (F
l
## ·F
## (k)
## ) +R
h
## (F
h
## ·F
## (k)
## )
## =F
## (k)
## + (R
l
## −R
h
## )D
## −1/2
## G
## AD
## −1/2
## G
## F
## (k)
## ,
## (9)
in  whichF
## (k)
## ={f
x
i,(k)
|i∈[1,N],x∈ {t,a,v}} ∈
## R
## |V
## G
## |×D
h
is the input at layerk.R
l
## ,R
h
## ∈R
## |V
## G
## |×|V
## G
## |
are
the weight matrices for low-frequency and high-frequency
information, respectively.  Eq. (9) can be written in another
form as
f
i,(k+1)
## =f
i,(k)
## +
## ∑
j∈N
i
r
l
ij
## −r
h
ij
## √
## |N
j
## |
## √
## |N
i
## |
f
j,(k)
## ,(10)
whereN
i
is  the  neighbouring  nodes  of  nodei.r
l
ij
and
r
h
ij
are the weight contributions of nodej’s low-frequency
and high-frequency signals to nodei, respectively, and they
meet the constraintr
l
ij
## +r
h
ij
## = 1.
To effectively learn the coefficientr
l
ij
## −r
h
ij
in Eq. (10),
we follow FAGCN [2] to employ a self-gating mechanism,
which  considers  the  correlation  between  the  central  node
and neighbours:
r
l
ij
## −r
h
ij
=tanh(W
## 3
## (f
i,(k)
## ⊕f
j,(k)
## )).(11)
Here,⊕is the concatenation operation andW
## 3
## ∈R
## 2D
h
## ×1
is a trainable weight matrix.  tanh(·) is the hyperbolic tan-
gent function that scales the value in[−1,1]. By this means,
the coefficientr
l
ij
## −r
h
ij
can readily model the varying im-
portance of different frequency constituents.  For instance,
ifr
l
ij
## −r
h
ij
<0,  the high-frequency messages dominate,
and nodeireceives the discrepancy between nodeiand the
neighbourj(i.e.,f
i,(k)
## −f
j,(k)
); and it holds vice versa.
Now we gradually spread the multi-frequency informa-
tion over the graph.  By stackingKlayers, each node re-
ceives the multi-frequency signals fromK-hop neighbours,
and we use outputs of the final layer as the multi-frequency
representations:
f
t
i
## =f
t
i,(K)
## ,
f
a
i
## =f
a
i,(K)
## ,
f
v
i
## =f
v
i,(K)
## .(12)
3.3.4    Differences with FAGCN
The graph learning rule of the above multi-frequency mod-
ule is closely related to Frequency Adaptation Graph Con-
volutional Networks (FAGCN) [2], which proposes to adap-
tively  integrate  low-frequency  and  high-frequency  signals
as well.  Although we derive inspiration from FAGCN, our
multi-frequency  module  contains  several  critical  distinc-
tions:  (i) FAGCN introduces a hyper-parameter to balance
the identity matrix and Laplacian matrix when defining fil-
ters while our method is hyper-parameter free; (ii) FAGCN
always  updates  node  embeddings  based  on  the  inputs  at
first layer, while we gradually refine the node embeddings
based on the outputs of previous layer. We present the per-
formance comparison between our multi-frequency module
and FAGCN in Section 5.5 and show by extensive experi-
ments that our design outperforms FAGCN.
3.4. Emotion classification
The  emotion  classifier  takes  as  input  the  concatenated
multivariate and multi-frequency representations to perform
emotion prediction. Mathematically,
e
i
## =
v
t
i
## ⊕
f
t
i
## ⊕
v
a
i
## ⊕
f
a
i
## ⊕
v
v
i
## ⊕
f
v
i
## ,(13)
wheree
i
is  the  emotion  representation  for  utterancei,
and  contains  both  multivariate  dependencies  and  multi-
frequency information.  Finally, we feede
i
into a softmax
layer to obtain the emotion category:
## ̃e
i
=ReLU(e
i
## ),
## P
i
=softmax(W
## 4
## ̃e
i
## +b
## 4
## ),
## ˆy
i
## =argmax
τ
## (P
i
## [τ]),
## (14)
## 10765

whereW
## 4
is trainable weight,P
i
## ∈R
## C
andˆy
i
is the pre-
dicted label for utteranceu
i
## .
3.5. Training objective
We follow prior works [18, 24] to use categorical cross-
entropy along withL
## 2
-regularization as the loss function:
## L=−
## 1
## ∑
## Num
s=1
c(s)
## Num
## ∑
i=1
c(i)
## ∑
j=1
logP
i,j
## [y
i,j
## ] +λ‖θ‖
## 2
## ,(15)
whereNumis the number of dialogues,c(i)is the number
of utterances in dialoguei,P
i,j
andy
i,j
are the probabilistic
distribution of class labels and the ground-truth label for ut-
terancejin dialoguei, respectively.λis theL
## 2
## -regularizer
weight andθdenotes the trainable parameters in the model.
## 4. Experiments
## 4.1. Datasets
We  compare  the  performance  of  our  proposed  M
## 3
## Net
against  prior  works  on  two  popular  multimodal  datasets,
IEMOCAP [3] and MELD [27],  following dominant data
split  protocol  and  modality  employment  as  in  previous
works [5, 17, 18].
IEMOCAPcontains 151 dyadic dialogues of ten speak-
ers and 7,433 utterances labelled with one of six emotion
categories: happy, sad, neutral, angry, excited, or frustrated.
We use 120 dialogues with 5,810 utterances for training and
validation,  and the rest for testing.   We employ language,
video and audio modalities for emotion prediction.
MELDis a multiparty emotional conversational dataset
which is collected from the TV showFriends. MELD con-
tains 1,433 rounds of conversations and 13,708 utterances.
Each utterance is annotated as one of seven emotion labels:
anger, disgust, sadness, joy, surprise, fear, or neutral.  We
use 1,039 dialogues with 9,989 utterances for training, 114
dialogues with 1,109 utterances for validation, and the rest
for testing.  We follow previous works [17, 18] to employ
language, video and audio modalities.
4.2. Unimodal feature extraction
In this paper, we use pre-extracted unimodal features fol-
lowing identical settings in previous studies [5, 11, 24].
The  textual  features  are  extracted  using  the  RoBERTa
Large  model  [22],  which  is  firstly  fine-tuned  for  emotion
prediction from the transcript of conversations.   After the
fine-tuning process, the utterances are fed to the model and
the  activations  from  the  final  four  layers  are  extracted  as
four textual vectors,  which are then normalized and aver-
aged for the final textual representation.  The dimension of
textual features in our paper is 1024.
The  acoustic  features  are  obtained  by  the  openSMILE
toolkit  [9].   The  visual  features  are  extracted  with  a  pre-
trained DenseNet [19] for the MELD dataset, and through a
DatasetBatchOptimizerD
h
LKDropout
IEMOCAP16Adam (lr=1e-4)512340.5
## MELD
16Adam (lr=1e-4)512330.4
Table 1. Details of hyper-parameters in our experiments.
3D-CNN for the IEMOCAP dataset. More details are stated
in appendix.
## 4.3. Baselines
For a comprehensive evaluation of M
## 3
Net, we compare
our model with the following state-of-the-art methods:
•CMN[14] seeks to model contextual information from
dialogue history.  It uses two GRUs for two speakers
and stores contexts as memories. It is not applicable to
multiparty scenarios, hence no results on MELD.
•ICON[13] is an extension of CMN, which connects
outputs  from  speaker  GRUs  in  CMN  with  another
GRU, so as to explicitly model inter-speaker interac-
tion. Similar to CMN, ICON is not applicable to mul-
tiparty scenarios, hence no results on MELD.
•DialogueRNN[24]  employs  three  GRU  cells  to  re-
spectively keep track of global context, speaker state
and  emotion  state  throughout  the  conversation.   It  is
capable of handling multiparty conversations.
•MetaDrop[5]  introduces  a  binary  maintain-or-drop
decision learning mechanism to learn adaptive fusion
paths,  as  well  as  simultaneously  capture  multimodal
and contextual relations.
•DialogueGCN[12] uses graph relational modelling to
encode  context.   Each  utterance  is  represented  as  a
node, and connected with other nodes in the same dia-
logue within a context window. It originally focuses on
textual modality and we extend it to multimodal sce-
nario by concatenating the unimodal embeddings.
•MMGCN[18]  constructs  a  heterogeneous  graph  by
regarding each modality of each utterance as a node. It
designs separate edge weighting mechanisms for inter-
modal and intra-modal edges, and encodes both multi-
modal and contextual information with deep layers.
•MM-DFN[17]  proposes  a  graph-based  dynamic  fu-
sion module to keep track of conversational context in
different semantic spaces, and enhance complementar-
ity between modalities.
4.4. Settings and evaluation metrics
The proposed model is implemented using PyTorch and
torch-geometric packages.   The networks are trained on a
machine with 1 NVIDIA GeForce RTX 3090.  We follow
dominant evaluation protocols to use accuracy and F1-score
as the metrics to measure the performance.  Paired t-test is
performed to test the significance of performance improve-
ment with a default significance level of 0.05.  Models are
## 10766

MethodsNetwork
IEMOCAP Average (w)MELD Average (w)
AccuracyF1AccuracyF1
GloVe
## CMN
## /
[14]Non-GNN-58.50--
## ICON
## ∗
[13]Non-GNN64.0063.50--
DialogueRNN
## †
[24]Non-GNN63.5162.9059.9257.60
MetaDrop
## ♦
[5]Non-GNN65.7665.58-58.30
DialogueGCN
## †
[12]GNN-based66.1766.2457.0155.59
## MMGCN
## †
[18]GNN-based65.8065.4160.4258.31
## MM-DFN
## †
[17]GNN-based68.2168.1859.8158.42
## M
## 3
Net (ours)GNN-based69.5069.0861.6559.22
RoBERTa
DialogueGCN
## †
[12]GNN-based63.9664.4463.4962.78
## MMGCN
## †
[18]GNN-based66.7966.9966.6365.13
DialogueRNN
## ♦
[24]Non-GNN68.6468.7265.9465.31
MetaDrop
## ♦
[5]Non-GNN69.3869.5966.6366.30
## MM-DFN
## †
[17]GNN-based69.8769.4867.0166.17
## M
## 3
Net (ours)GNN-based72.4672.4968.2867.05
Table 2.  Comparison with previous state-of-the-art methods on IEMOCAP and MELD. Bold font denotes the best performances.  Aver-
age(w) = weighted average.
## /
from [24];
## ∗
from [13];
## ♦
from [5];
## †
from our reimplementation using open source codes.
trained using Adam [21] with a batch size of 16 on both
datasets.   We testLandKin the range from 1 to 7 and
present the best-performing results.  Full details of hyper-
parameters for both datasets are shown in Table 1.   Code
is available athttps://github.com/feiyuchen7/
## M3NET.
- Results and analysis
5.1. Comparison with state-of-the-arts
We  contrast  our  model  with  a  wide  range  of  state-of-
the-art  methods  in  Table  2.   It  can  be  seen  that  on  both
datasets, our proposed M
## 3
Net surpasses previous methods
and achieves new state-of-the-art records in terms of both
metrics  of  accuracy  and  F1-score.   In  particular,  M
## 3
## Net
outperforms previous GNN-based methods,  including Di-
alogueGCN,  MMGCN  and  MM-DFN,  which  manually
tweak edge weighting strategies with complicated relation
learning  or  similarity  metrics,  to  capture  multimodal  and
contextual relationships.  We suggest that the advantage of
our method is due to the investigation into multivariate and
multi-frequency information among modalities and context,
which is neglected by previous methods.
5.2. Textual features from BERT vs. GloVe
As stated in Section 4.2, in this work the inputting textual
features are extracted from a pre-trained RoBERTa Large
model, which according to our observation, can boost up the
performance compared with traditional GloVe-based textual
features.  In order to verify whether our model can deliver
good performance regardless of the sources of textual fea-
tures, we further conduct experiments using GloVe embed-
dings and present comparison with previous methods.  The
results are shown in Table 2. It can be observed that M
## 3
## Net
## Methods
## IEMOCAPMELD
Acc.F1Acc.F1
## M
## 3
## Net72.4672.4968.2867.05
1w/o multivariate info.70.0670.0567.7466.36
2w/o multi-frequency info.69.8769.7467.3666.03
3w/o hyperedge weightω(e)70.3070.4568.1166.99
## 4
w/o node weightγ
e
## (v)70.9871.0268.0566.92
5w/o both weights70.1270.0967.8966.75
6H→Gin series68.3968.4468.2066.84
7G →Hin series69.5069.7068.0566.85
Table 3. Ablation studies of M
## 3
## Net.
outperforms other baselines based on either textual feature
source, through which we can infer that our multivariate and
multi-frequency modelling delivers major improvements.
5.3. Ablation studies
To gain better insights to the constituents of our model,
we  perform  ablation  studies  on  the  key  components  of
## M
## 3
Net and present results in Table 3.
Effect  of  multivariate  information.We  first  explore
the effect of multivariate information among modalities and
context. To achieve this, we remove the multivariate propa-
gation module (i.e.,  the hypergraphH) and perform clas-
sification  based  on  multi-frequency  representations  only,
shown as variant 1 in Table 3.  Under this setting, we can
observe a decrease of 2.40% in accuracy and 2.44% in F1-
score on IEMOCAP, as well as a decrease of 0.54% in accu-
racy and 0.69% in F1-score on MELD. This demonstrates
the  effectiveness  of  introducing  multivariate  propagation,
which can naturally encode relationships of higher arity.
Effect of multi-frequency information.Another core
component  of  M
## 3
Net  is  the  multi-frequency  propagation
module. Similarly, we test the importance of this module by
## 10767

(a) Effect of L on IEMOCAP(b) Effect of K on IEMOCAP
(c) Effect of L on MELD(d) Effect of K on MELD
## 67
## 68
## 69
## 70
## 71
## 72
## 73
## 1234567
Va l u e   o f   L
## IEMOCAP
Accur acyF1
## 67
## 68
## 69
## 70
## 71
## 72
## 73
## 1234567
Va l u e   o f   K
## IEMOCAP
Accur acyF1
## 65
## 65.5
## 66
## 66.5
## 67
## 67.5
## 68
## 68.5
## 1234567
Value of L
## MELD
Accur acyF1
## 65
## 65.5
## 66
## 66.5
## 67
## 67.5
## 68
## 68.5
## 1234567
Value of K
## MELD
Accur acyF1
Figure 3.  Results of M
## 3
Net at different graph layers.  In (a) and (c), effects ofLare tested by fixingKas in the best-performing models.
In (b) and (d), effects ofKare tested by fixingLas in the best-performing models.
having it removed and performing predictions using multi-
variate  representations  only.   Variant  2  reports  the  results
of  this  configuration,  from  which  a  sharp  degradation  of
performance can be observed.  This stands as a convincing
proof of the validity of introducing different frequency in-
formation into ERC, which can guide the model to capture
the varying importance of emotion discrepancy and emotion
commonality in the local neighbourhood.
Effect of weights in the hypergraph.In Section 3.2.1,
we define two types of weights in hypergraphHto capture
the  multivariate  relationships  in  a  fine-grained  level.   We
hence conduct experiments to verify the effect of these two
weights.  It can be seen from variants 3 to 5 that removing
either or both weights (i.e., setting weight valueω(e)or/and
γ
e
(v)as 1) harms the performance on both datasets.  This
indicates that the formulated weights benefit the training.
Effect  of  parallel  modelling.In  M
## 3
Net,  we  propa-
gate  multivariate  and  multi-frequency  information  in  par-
allel.   We further conduct experiments to compare it with
two-step serial modelling and show the results as variants
6 and 7.  Serial modelling slightly reduces the performance
on MELD but leads to dramatic decreases on IEMOCAP,
which implies the effectiveness of the parallel modelling.
5.4. Discussions on graph layers
## M
## 3
Net contains two parallel graphs, and the graph prop-
agation plays a pivotal role.   To investigate the impact of
stacking different graph layers, we conduct a grid search on
the number of layers. Specifically, we search the layer num-
bers  of  multivariate  propagation  (L)  and  multi-frequency
propagation (K) in the range from 1 to 7 and summarize
the results in Figure 3. On IEMOCAP, the effects ofLand
Kare similar. At first, the results steadily improve as stack-
ing more layers, and peak atL= 3andK= 4respectively.
Further stacking more layers has little positive impact on the
performance.  On the other hand, it can be noticed that the
results on MELD are less sensitive to the number of graph
layers, with no special pattern, as shallow or deep layers can
all yield decent performance.
## 60
## 63
## 66
## 69
## 72
## 75
## 00.10.20.30.40.50.60.70.80.91
Va l u e   o f   ε
## IEMOCAP
ACC. w/ FAGCNF1  w/ FAGCN
ACC. - oursF1 - ours
## 63
## 64
## 65
## 66
## 67
## 68
## 69
## 00.10.20.30.40.50.60.70.80.91
Value of ε
## MELD
ACC. w/ FAGCNF1  w/ FAGCN
ACC. - oursF1 - ours
Figure 4. Performance comparison with FAGCN.
5.5. Comparison with FAGCN
As stated in Section 3.3.4, the graph propagation rule of
our  multi-frequency  module  is  closely  related  to  FAGCN
[2] but retains critical distinctions.  To further demonstrate
the effectiveness of our method,  we present an additional
comparison  with  FAGCN.  Specifically,  we  maintain  the
multivariate module, and replace our multi-frequency mod-
elling strategy (Eq. (7) to Eq. (11)) with the one introduced
in  FAGCN.  Since  FAGCN  introduces  a  hyper-parameter
∈[0,1]when defining filters,  we testin the range of
[0,1]with a step of 0.1.  The comparison is summarized in
Figure 4. Apparently,is a vital factor and dramatically im-
pacts the performance, especially on IEMOCAP. However,
under no circumstances can these variants with FAGCN out-
perform the original M
## 3
Net.  This indicates the superiority
of our multi-frequency modelling mechanism.
## 6. Conclusion
This paper proposes a GNN-based model to address the
ERC  problem.   We  present  Multivariate  Multi-frequency
Multimodal Graph Neural Network (M
## 3
Net) to investigate
the  multivariate  relationships  among  modalities  and  con-
text,  and take full advantage of different frequency infor-
mation which reflects emotion discrepancy and commonal-
ity. Extensive experimental results on two datasets show the
superiority of our model.
## 10768

## References
[1]  Song Bai, Feihu Zhang, and Philip H. S. Torr.  Hypergraph
convolution  and  hypergraph  attention.Pattern  Recognit.,
## 110:107637, 2021. 2, 4
[2]  Deyu Bo, Xiao Wang, Chuan Shi, and Huawei Shen. Beyond
low-frequency information in graph convolutional networks.
InThirty-Fifth  AAAI  Conference  on  Artificial  Intelligence,
AAAI 2021, pages 3950–3957, 2021. 2, 4, 5, 8
[3]  Carlos   Busso,Murtaza   Bulut,Chi-Chun   Lee,Abe
## Kazemzadeh,  Emily  Mower,  Samuel  Kim,  Jeannette  N.
Chang, Sungbok Lee, and Shrikanth S. Narayanan.  IEMO-
CAP: interactive emotional dyadic motion capture database.
## Lang. Resour. Evaluation, 42(4):335–359, 2008. 2, 6
## [4]  Feiyu Chen, Jie Shao, Anjie Zhu, Deqiang Ouyang, Xueliang
Liu, and Heng Tao Shen.  Modeling hierarchical uncertainty
for multimodal emotion recognition in conversation.IEEE
## Trans. Cybern., 2022. 2
## [5]  Feiyu  Chen,  Zhengxiao  Sun,  Deqiang  Ouyang,  Xueliang
Liu, and Jie Shao.  Learning what and when to drop:  Adap-
tive multimodal and contextual dynamics for emotion recog-
nition in conversation.  InMM ’21:  ACM Multimedia Con-
ference, pages 1064–1073, 2021. 1, 2, 6, 7
[6]  Hyung-Gun   Chi,   Myoung   Hoon   Ha,   Seung-geun   Chi,
Sang  Wan  Lee,  Qixing  Huang,  and  Karthik  Ramani.    In-
fogcn:   Representation  learning  for  human  skeleton-based
action recognition.  InIEEE/CVF Conference on Computer
Vision and Pattern Recognition, CVPR 2022, pages 20154–
## 20164, 2022. 3
[7]  Uthsav Chitra and Benjamin J. Raphael.  Random walks on
hypergraphs  with  edge-dependent  vertex  weights.   InPro-
ceedings of the 36th International Conference on Machine
Learning, ICML 2019, pages 1172–1181, 2019. 2, 4
[8]  Yushun Dong, Kaize Ding, Brian Jalaian, Shuiwang Ji, and
Jundong Li.  Adagnn:  Graph neural networks with adaptive
frequency response filter.  InCIKM ’21:  The 30th ACM In-
ternational Conference on Information and Knowledge Man-
agement, pages 392–401, 2021. 2
## [9]  Florian  Eyben,  Martin  W
## ̈
ollmer,  and  Bj
## ̈
orn  W.  Schuller.
Opensmile: the munich versatile and fast open-source audio
feature extractor.   InProceedings of the 18th International
Conference on Multimedia 2010, pages 1459–1462, 2010. 6
[10]  Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and
Yue Gao.  Hypergraph neural networks.  InThe Thirty-Third
AAAI  Conference  on  Artificial  Intelligence,   AAAI  2019,
pages 3558–3565, 2019. 2
## [11]  Deepanway Ghosal, Navonil Majumder, Alexander F. Gel-
bukh, Rada Mihalcea, and Soujanya Poria.  COSMIC: com-
monsense  knowledge  for  emotion  identification  in  conver-
sations.   InFindings of the Association for Computational
Linguistics: EMNLP 2020, pages 2470–2481, 2020. 6
## [12]  Deepanway  Ghosal,  Navonil  Majumder,  Soujanya  Poria,
Niyati  Chhaya,  and  Alexander  F.  Gelbukh.   Dialoguegcn:
A graph convolutional neural network for emotion recogni-
tion in conversation. InProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural Language
Processing, EMNLP-IJCNLP 2019, pages 154–164, 2019. 1,
## 3, 4, 6, 7
## [13]  Devamanyu Hazarika, Soujanya Poria, Rada Mihalcea, Erik
Cambria, and Roger Zimmermann.  ICON: interactive con-
versational memory network for multimodal emotion detec-
tion.   InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, pages 2594–2604,
## 2018. 1, 2, 6, 7
## [14]  Devamanyu  Hazarika,  Soujanya  Poria,  Amir  Zadeh,  Erik
Cambria, Louis-Philippe Morency, and Roger Zimmermann.
Conversational memory network for emotion recognition in
dyadic dialogue videos.  InProceedings of the 2018 Confer-
ence of the North American Chapter of the Association for
## Computational Linguistics: Human Language Technologies,
NAACL-HLT  2018,  Volume  1  (Long  Papers),  pages  2122–
## 2132, 2018. 2, 6, 7
[15]  Devamanyu  Hazarika,  Roger  Zimmermann,  and  Soujanya
Poria.   MISA:  modality-invariant  and  -specific  representa-
tions for multimodal sentiment analysis.   InMM ’20:  The
28th  ACM International  Conference on  Multimedia,  pages
## 1122–1131, 2020. 1
[16]  Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong
Zhang, and Meng Wang.   Lightgcn:  Simplifying and pow-
ering  graph  convolution  network  for  recommendation.   In
Proceedings of the 43rd International ACM SIGIR confer-
ence on research and development in Information Retrieval,
SIGIR 2020, pages 639–648, 2020. 3
[17]  Dou Hu, Xiaolong Hou, Lingwei Wei, Lian-Xin Jiang, and
Yang Mo.  MM-DFN: multimodal dynamic fusion network
for emotion recognition in conversations.  InIEEE Interna-
tional Conference on Acoustics, Speech and Signal Process-
ing, ICASSP 2022, pages 7037–7041, 2022. 2, 3, 6, 7
[18]  Jingwen  Hu,   Yuchen  Liu,   Jinming  Zhao,   and  Qin  Jin.
MMGCN:  multimodal  fusion  via  deep  graph  convolution
network  for  emotion  recognition  in  conversation.   InPro-
ceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing, ACL/IJCNLP
2021, (Volume 1: Long Papers), pages 5666–5675, 2021.  1,
## 2, 3, 4, 6, 7
[19]  Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kil-
ian  Q.  Weinberger.   Densely  connected  convolutional  net-
works.  In2017 IEEE Conference on Computer Vision and
Pattern Recognition, CVPR 2017, pages 2261–2269, 2017.
## 6
[20]  Yuchi Huang, Qingshan Liu, and Dimitris N. Metaxas. Video
object segmentation by hypergraph cut. In2009 IEEE Com-
puter  Society  Conference  on  Computer  Vision  and  Pattern
Recognition (CVPR 2009), pages 1738–1745, 2009. 2
[21]  Diederik P. Kingma and Jimmy Ba.   Adam:  A method for
stochastic optimization.  In3rd International Conference on
Learning Representations, ICLR 2015, 2015. 7
## [22]  Yinhan Liu,  Myle Ott,  Naman Goyal,  Jingfei Du,  Mandar
## Joshi,  Danqi Chen,  Omer Levy,  Mike Lewis,  Luke Zettle-
moyer, and Veselin Stoyanov. Roberta: A robustly optimized
BERT pretraining approach.CoRR, abs/1907.11692, 2019.
## 6
## 10769

[23]  Sijie  Mai,  Ying  Zeng,  Shuangjia  Zheng,  and  Haifeng  Hu.
Hybrid  contrastive  learning  of  tri-modal  representation  for
multimodal sentiment analysis.IEEE Trans. Affect. Comput.,
## 2022. 1
## [24]  Navonil Majumder,  Soujanya Poria,  Devamanyu Hazarika,
Rada  Mihalcea,  Alexander  F.  Gelbukh,  and  Erik  Cambria.
Dialoguernn:   An  attentive  RNN  for  emotion  detection  in
conversations.  InThe Thirty-Third AAAI Conference on Ar-
tificial Intelligence, AAAI 2019, pages 6818–6825, 2019.  1,
## 2, 6, 7
## [25]  Yuzhao  Mao,  Qi  Sun,  Guang  Liu,  Xiaojie  Wang,  Weiguo
Gao, Xuan Li, and Jianping Shen.  Dialoguetrm:  Exploring
the intra- and inter-modal emotional behaviors in the conver-
sation.CoRR, abs/2010.07637, 2020. 2
[26]  Hoang  NT  and  Takanori  Maehara.   Revisiting  graph  neu-
ral  networks:    All  we  have  is  low-pass  filters.CoRR,
abs/1905.09550, 2019. 2, 4
## [27]  Soujanya Poria,  Devamanyu Hazarika,  Navonil Majumder,
Gautam Naik, Erik Cambria, and Rada Mihalcea.   MELD:
A multimodal multi-party dataset for emotion recognition in
conversations.  InProceedings of the 57th Conference of the
Association for Computational Linguistics, ACL 2019, Vol-
ume 1: Long Papers, pages 527–536, 2019. 2, 6
[28]  Weizhou Shen, Siyue Wu, Yunyi Yang, and Xiaojun Quan.
Directed acyclic graph network for conversational emotion
recognition.  InProceedings of the 59th Annual Meeting of
the Association for Computational Linguistics and the 11th
International  Joint  Conference  on  Natural  Language  Pro-
cessing, ACL/IJCNLP 2021, (Volume 1: Long Papers), pages
## 1551–1560, 2021. 1, 3
## [29]  David I. Shuman, Sunil K. Narang, Pascal Frossard, Anto-
nio Ortega,  and Pierre Vandergheynst.   The emerging field
of signal processing on graphs: Extending high-dimensional
data analysis to networks and other irregular domains.IEEE
## Signal Process. Mag., 30(3):83–98, 2013. 5
## [30]  Xiangguo Sun, Hongzhi Yin, Bo Liu, Hongxu Chen, Jiuxin
Cao, Yingxia Shao, and Nguyen Quoc Viet Hung.  Hetero-
geneous hypergraph embedding for graph classification.  In
WSDM ’21, The Fourteenth ACM International Conference
on Web Search and Data Mining, pages 725–733, 2021. 2
## [31]  Felix Wu, Amauri H. Souza Jr., Tianyi Zhang, Christopher
Fifty, Tao Yu, and Kilian Q. Weinberger.  Simplifying graph
convolutional networks.  InProceedings of the 36th Interna-
tional Conference on Machine Learning, ICML 2019, pages
## 6861–6871, 2019. 2, 4
[32]  Liwen  Xu,   Zhengtao  Wang,   Bin  Wu,   and  Simon  Lui.
MDAN: multi-level dependent attention network for visual
emotion  analysis.   InIEEE/CVF  Conference  on  Computer
Vision and Pattern Recognition,  CVPR 2022,  pages 9469–
## 9478, 2022. 1
[33]  Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu.  Learning
modality-specific representations with self-supervised multi-
task learning for multimodal sentiment analysis.  InThirty-
Fifth AAAI Conference on Artificial Intelligence, AAAI 2021,
pages 10790–10797, 2021. 1
## [34]  Dong Zhang, Weisheng Zhang, Shoushan Li, Qiaoming Zhu,
and Guodong Zhou. Modeling both intra- and inter-modal in-
fluence for real-time emotion detection in conversations.  In
MM ’20: The 28th ACM International Conference on Multi-
media, pages 503–511, 2020. 2
## 10770