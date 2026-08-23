# Ada2I: Enhancing Modality Balance for Multimodal Conversational Emotion Recognition

[Cam-Van Thi Nguyen](https://orcid.org/0009-0001-9675-2105)

[The-Son Le](https://orcid.org/0009-0006-7639-129X)

VNU University of Engineering and Technology [URL 🔗](https://orcid.org/0009-0001-9675-2105)

VNU University of Engineering and Technology

Hanoi, Vietnam vanntc@vnu.edu.vn

Hanoi, Vietnam

21020089@vnu.edu.vn

## [Anh-Tuan Mai](https://orcid.org/0009-0002-7211-2637)

VNU University of Engineering and Technology

Hanoi, Vietnam 20020269@vnu.edu.vn

## ABSTRACT

Multimodal Emotion Recognition in Conversations (ERC) is a typi- cal multimodal learning task in exploiting various data modalities concurrently. Prior studies on effective multimodal ERC encounter challenges in addressing modality imbalances and optimizing learn- ing across modalities. Dealing with these problems, we present a novel framework named Ada2I, which consists of two inseparable modules namely Adaptive Feature Weighting (AFW) and Adaptive Modality Weighting (AMW) for feature-level and modality-level balancing respectively via leveraging both Inter- and Intra-modal interactions. Additionally, we introduce a refined disparity ratio as part of our training optimization strategy, a simple yet effective measure to assess the overall discrepancy of the model’s learning process when handling multiple modalities simultaneously. Exper- imental results validate the effectiveness of Ada2I with state-of- the-art performance compared to baselines on three benchmark datasets, particularly in addressing modality imbalances.

## CCS CONCEPTS

• Information systems → Sentiment analysis; • Computing

methodologies → Discourse, dialogue and pragmatics.

## KEYWORDS

Multimodal Emotion Recognition, Imbalance Modality, Adaptive Feature Weighting, Adaptive Modality Weighting, Disparity ratio

## ACM Reference Format:

Cam-Van Thi Nguyen, The-Son Le, Anh-Tuan Mai, and Duc-Trong Le. 2024. Ada2I: Enhancing Modality Balance for Multimodal Conversational Emo- tion Recognition. In Proceedings ofthe 32nd ACM International Conference on Multimedia (MM ’24), October 28-November 1, 2024, Melbourne, VIC, Aus- traliaProceedings ofthe 32nd ACM International Conference on Multimedia (MM’24), October 28-November 1, 2024, Melbourne, Australia. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3664647.3681648 [URL 🔗](https://doi.org/10.1145/3664647.3681648)

[https://doi.org/10.1145/3664647.3681648](https://doi.org/10.1145/3664647.3681648)

## [Duc-Trong Le](https://orcid.org/0000-0003-4621-8956)

VNU University of Engineering and Technology [URL 🔗](https://orcid.org/0000-0003-4621-8956)

Hanoi, Vietnam trongld@vnu.edu.vn

(b)

*Figure 1: (a) Weighted F1 scores for the multimodal set- ting (T+A+V) compared with each unimodal encoder, and (b) batch-average unimodal-logit scores.*

## 1 INTRODUCTION

Multimodal learning is an approach to building models that can pro- cess and integrate information from multiple heterogeneous data modalities [2, 20, 21], including image, text, audio, video, and table. Since numerous tasks in the real world involve multiple modalities, multimodal learning has become increasingly important and at- tracted widespread attention as an effective way to accomplish these tasks. In recent years, the field of Emotion Recognition in Conver- sations (ERC) has witnessed a surge in effective models [8, 27, 30]. Moving beyond unimodal recognition, the utilization ofmultimodal data offers a multidimensional perspective for more nuanced emo- tion discernment [9, 19, 24]. Consequently, the incorporation of multimodal data is a natural evolution for enhancing emotion recog- nition in conversations. However, the widespread adoption ofmulti- modal learning has revealed underlying challenges, with a primary focus on modality imbalances. These imbalances entail disparities in the contributions of individual modalities to the final decision- making process. [URL 🔗](#page-0)

As illustrated in Figure. 1, the text modality quickly addresses the overall model performance and the joint logit scores, whereas [URL 🔗](#page-0)


the visual and audio modalities remain under-optimized through- out the training process. In addressing modality imbalance, diverse terminologies have emerged to characterize this phenomenon and explore its underlying causes. Terms such as “greedy nature” [39], “modality collapse” [15], and “modality imbalance” [6, 22] have been employed in various studies. These terms are associated with factors such as the “suppression of dominant modalities” [26], “different convergence rates” [36], “diminishing modal marginal utility” [37], or “modality competition” [14]. In essence, two primary perspec- tives emerge regarding this problem [37]: firstly, modalities exhibit varying levels of dominance, with models often overly reliant on a dominant modality with the highest convergence speed, thereby impeding the full utilization ofother modalities with slower conver- gence speeds. Secondly, modal encoder optimization varies, necessi- tating the adoption of multiple strategies. Some approaches [7, 26] attempt to modulate the learning rates of different modalities based on the fusion modality. However, these approaches often overlook the impact ofintra-modal data enhancement [46]. For instance, right from the initial representations through the modal encoder, the out- puts can lead to misleading final results, resulting in its weakened position across all modalities. Hence, from the outset, it is crucial to enhance representations for each modality, regardless of whether they are weak or strong, as it can affect the imbalance in learning across modalities. [URL 🔗](#page-0)

Moreover, current methodologies primarily focus on interactions between pairs ofmodalities [6, 22, 26, 40], resulting in complex com- putations and inadequate treatment across all modalities. These methods are commonly applied in tasks such as audio-visual learn- ing [26, 40] and multimodal affective computing [46], often using datasets related to sarcasm detection, sentiment analysis, or humor detection. However, there is a lack of methods explicitly tailored for multimodal ERC tasks, especially for well-known multimodal datasets like IEMOCAP [3], MELD [28], and CMU-MOSEI [1]. Ad- ditionally, in recent prominent studies [17, 33], while overall per- formance for multimodal ERC tasks has notably increased, a closer examination of the “importance ofmodality” reveals that pairwise modalities consistently fail to achieve satisfactory performance, cre- ating a significant gap compared to leveraging all three modalities simultaneously. Therefore, it is crucial to simultaneously leverage learning from all modalities while also significantly enhancing the capabilities ofweaker modalities to improve the overall learning performance of multimodal ERC models in practical applications. [URL 🔗](#page-0)

In this paper, we propose a novel framework named Ada2I that addresses imbalances in learning across audio, text, and visual modalities for multimodal ERC. It consists of two primary mod- ules including Adaptive Feature Weighting (AFW) and Adaptive Modality Weighting (AMW) for feature-level and modality-level bal- ancing respectively in the consideration of Inter- and Intra-modal interactions. Focusing on feature-level balancing using Adaptive Feature Weighting (AFW), we apply tensor contraction to infer feature-aware attention weights for each modality, which aims to produce a feature-level balanced representation for each conversa- tion. As an important component of AFW, Attention Mapping Net- work controls the balancing via maximizing the alignment between unimodal features and their corresponding attention coefficients. For modality-level balancing using Adaptive Modality Weighting (AMW), we further exploit feature-level balanced representations

from the preceding AFW module to generate modality-level bal- anced ones through modality-wise normalization of features and learning weights before being used to enhance the emotion recogni- tion. Additionally, we utilize the concept ofdisparity ratio, although with modifications compared to the study by Peng et al. [26], called OGM-GE, as a value to supervise the training process and evaluate the model. Specifically, while OGM-GE [26] introduced gradient modulation for pairs ofmodalities, we refine it to handle all three modalities simultaneously—textual, visual, and audio—in the Multi- modal emotion recognition in conversation task. This adjustment reduces model complexity and overall processing time, leading to enhanced efficiency. To summarize, our contributions are as follows: [URL 🔗](#page-0)

- We propose an end-to-end framework named Ada2I that addresses the issue of imbalance learning across modali- ties comprehensively for the multimodal ERC task. It not only considers modality-level imbalances but also leverages feature-level representations to contribute to the balancing step in the learning process.

- With two modules intricately designed yet inseparable, Adap- tive Feature Weighting (AFW) is crafted to enhance the rep- resentation of each conversation at the feature level, while Adaptive Modality Weighting (AMW) is proposed to opti- mize the modality-level learning weights during training. Additionally, we redefine the disparity ratio, a simple yet effective measure, to assess the overall discrepancy of the model’s learning process when simultaneously handling mul- tiple modalities, rather than just two as in the original ap- proach from Peng et al. [26]. [URL 🔗](#page-0)

- Our empirical experiments illustrate the effectiveness and enhancements of Ada2I in comparison to existing state-of- the-art approaches dealing with modality imbalance across three prevalent multimodal ERC datasets including IEMO- CAP [3], MELD [28], and CMU-MOSEI [1]. [URL 🔗](#page-0)

## 2 RELATEDWORK

## 2.1 Multimodal Emotion Recognition

Multimodal Emotion Recognition (ERC) has emerged as a focal point within the affective computing community, garnering sig- nificant attention in recent years. The integration of multimodal data provides a multidimensional perspective, enabling a more nuanced understanding of emotions. Moreover, researchers have increasingly turned to multimodal fusion techniques, combining text, audio, and visual cues to enhance multimodal ERC perfor- mance [9, 10, 16, 18, 24, 25]. ICON [9] employs two Gated Recur- rent Units (GRUs) to capture speaker information, supplemented by global GRUs to track changes in emotional states throughout con- versations. Similarly, MMGCN [38] utilizes Graph Convolutional Networks (GCNs) to capture contextual information, effectively leveraging multimodal dependencies and speaker information. On the other hand, Multilogue-Net [31] introduces a solution utilizing a context-aware RNN and employing pairwise attention as a fu- sion mechanism. TBJE [4], adopts a transformer-based architecture with modular co-attention to jointly encode multiple modalities. Additionally, COGMEN [16] is a multimodal context-based graph neural network that integrates both local (speaker information) and global (contextual information) aspects of conversation. Moreover, [URL 🔗](#page-0)


CORECT [24] employs relational temporal Graph Neural Networks (GNNs) with cross-modality interaction support, effectively captur- ing conversation-level interactions and utterance-level temporal re- lations. GraphMFT [18] utilizes multiple enhanced graph attention networks to capture intra-modal contextual information and inter- modal complementary information. More recently, DF-ERC [17] emphasizes both feature disentanglement and fusion while tak- ing into account both multimodalities and conversational contexts. Moreover, AdaIGN [33] employs the Gumbel Softmax trick to adap- tively select nodes and edges, enhancing intra- and cross-modal interactions. While these methods primarily focus on designing model structures, they overlook the challenges posed by modality imbalance during multimodal learning. [URL 🔗](#page-0)

## 2.2 Imbalanced multimodal learning

Despite the suggestion by [13] that integrating multiple modalities could enhance the accuracy of latent space estimations, thereby im- proving the efficacy ofmultimodal models, our investigation within the multimodal ERC task reveals a phenomenon contradicting this notion. The problem ofmodality imbalance persists as a significant challenge in multimodal learning frameworks involving low-quality data [43], particularly in tasks such as multimodal ERC. Conven- tional methods often prioritize one modality over others, assuming that certain types of sensory data are more relevant for a given task. For example, textual cues may receive greater emphasis, while visual or audio cues alone might be prioritized [16, 24, 38]. Current methodologies for addressing imbalanced multimodal learning pri- marily focus on tasks such as audio-visual learning with a focus on optimizing pairwise modality learning [6, 26, 40], sentiment analysis, and sarcasm detection [46]. However, these approaches often have task-specific limitations and framework restrictions, limiting their broader applicability. For instance, Wang et al. [36] identified that different modalities overfit and generalize at different rates, leading to suboptimal solutions when jointly trained using a unified optimization strategy. Peng et al. [26] proposed OGM-ME method where the better-performing modality dominates the gra- dient update, suppressing the learning process of other modalities. MMCosine [40] employs normalization techniques on features and weights to promote balanced and improved fine-grained learning across multiple modalities. Notably, there is a lack of specific ap- proaches tailored for multimodal ERC apart from the work byWang et al. [37]. Recently, Wang et al. [37] observed a phenomenon re- ferred to as “diminishing modal marginal utility” and proposed fine- grained adaptive gradient modulation, which was applied to ERC, while I2MCL considers both data difficulty and modality balance for multimodal learning based on curriculum learning for affective computing, though not specifically for emotion recognition. To comprehensively address the challenge ofmodality imbalance in multimodal ERC, we propose an end-to-end model that ensures balance among text, audio, and visual modalities during training.

## 3 METHODOLOGY

## 3.1 Preliminary

3.1.1 Tensor Ring Decomposition.

sions)

tensors of order 3:

A tensor of order 𝐾 (𝐾 dimen-

T ∈

R𝑑1×𝑑2×...×𝑑𝐾

can be represented as a sequence of core

G𝑗 ∈ R𝑑𝑗 ×𝑟𝑗 ×𝑟𝑗+1

, where the last core tensor has

the form G𝐾 ∈ R𝑑𝐾×𝑟𝐾×𝑟1 . The dimensions 𝑟1, 𝑟2, ..., 𝑟𝑘 are called tensor ranks. In that case, T is represented in the form of a tensor ring 𝑇𝑟{G1, G2, ..., G𝑘 } as follows: T = G1 ×2 3 G2 ×2 4 ... ×2 𝑘+2 G𝑘 . In

which ×𝑚 denotes the tensor contraction operation with mode-

𝑛

(𝑚 𝑛 ). For example, with G1 ∈ R𝑑1×𝑟1×𝑟2 , G2 ∈ R𝑑2×𝑟2×𝑟3 and

G3 ∈ R𝑑3×𝑟3×𝑟1 , T is represented as G1 ×2 3 G2 ×2 4 G3 ∈ R𝑑1×𝑑2×𝑑3 .

3.1.2 Problem Definition. In the context of a conversation 𝐶 with 𝑁 utterances {𝑢1, 𝑢2, . . . , 𝑢𝑁}, the task of Emotion Recognition in Conversations (ERC) is to predict the emotion label for each utterance in the conversation from a predefined emotion category set E. Each utterance is associated with 𝑀 modalities, i.e. textual (t), audio (a), and visual (v) modalities, represented as:

where 𝑢𝑖 ∈ R𝑀×𝑑, 𝑑 signifies the dimension of modal features. For each modality 𝑚, we derive multimodal features {X𝑚}𝑚∈{𝑡,𝑎,𝑣} ∈ R𝑑𝑚×𝑁 for the conversation 𝐶. Here, {𝑑𝑚}𝑚∈{𝑡,𝑎,𝑣} is the feature dimension of each modality.

In the following sub-section, we outline our proposed model Ada2I, including its main sub-modules: (1) Modality Encoder, (2) Adaptive Feature Weighting and (3) Adaptive ModalityWeighting. We also refine the disparity ratio metric as part of our Training Optimization Strategy. Figure 2 illustrates architecture ofAda2I. [URL 🔗](#page-0)

## 3.2 Modality Encoder

Given a conversation 𝐶, a Transformer [34] network is utilized as the encoder to generate a unimodal representation Z𝑚 ∈ R𝑁×𝑑𝑚 respecting to the modality 𝑚 as: [URL 🔗](#page-0)

where the function 𝜙 (𝜃 (𝑚) ) is the Transformer network with learn- able parameter 𝜃 (𝑚) .

## 3.3 Adaptive Feature Weighting (AFW)

3.3.1 Tensor-based Multimodal Interaction Representation. Moti- vated by the tensor-ring decomposition method introduced by [44], we extend the traditional attention mechanism by replacing the query (Q) and key (K) representations with tensor-ring decomposition- based counterparts. This modification results in query tensor-ring representation G𝑄 and key tensor-ring representation G𝐾, which facilitate the acquisition of more compact modality representa- tions. Additionally, inspired by [32], we integrate a tensor-based multi-way interaction transformer architecture into our model. This enhancement allows the model to capture multi-way interac- tions among modalities, thereby enhancing its capability to discern intricate multimodal relationships. [URL 🔗](#page-0)

We employ a tensor-ring-based generation function to retrieve the multi-interaction multimodal query tensor Q and key tensor K from the input modality presentations Z𝑚. Specifically, we compute Q and K as follows:


ℎ 𝜃𝑡 𝑚

𝑋𝑣

𝑋𝑎

𝑋𝑡

Gradient modulation

~𝒩 0, ∑ 𝑠𝑔𝑑(𝜃𝑡 𝑚 )

𝑍𝑘 𝑣

𝑍𝑘

ො𝑔(𝜃𝑡 𝑚 )

𝜌𝑡 𝑚

𝑘𝑡 𝑚.

⊛

Backward

*Figure 2: Illustration of Ada2I framework*

Here, Tr{.} represents the tensor-ring decomposition function, which

naturally provides the low-rank core tensor representations G𝑚

and G𝑚 for each modality.

𝐾

To perform multimodal attention in the tensor space, we need to compute the attention coefficient matrix, Θ, from the tensorized input. To achive this, we can first compute the Tensor-ring Key representation and Tensor-ring Query representation of input data,

G𝑚 𝑄 ∈ R𝑑𝑚×𝑟𝑠 ×𝑟𝑤 and G𝑚 𝐾 ∈ R𝑑𝑚×𝑟𝑠 ×𝑟𝑤 , where 𝑚 ∈ {𝑡, 𝑎, 𝑣}, the

index 𝑠, 𝑤 ∈ {1, 2, 3}, and 𝑠 ≠ 𝑤. The attention coefficient matrix Θ of modality 𝑚 is formulated as follows:

𝑄

3.3.2 Adaptive Feature Weighting (AFW). This module addresses the varying impact of each modality on inter-modality and intra- modality interactions using attention mechanism. First, we calcu- late the attention pooling matrices A(𝑚) ∈ R𝑟𝑠 ×𝑟𝑤 by averaging Θ(𝑚) across the modality dimension 𝑑𝑚, 𝑚 ∈ {𝑡, 𝑎, 𝑣}. Inspired by MMT [32], the feature-aware attention matrix 𝐴𝑡𝑡𝑚 ∈ R𝑁×𝑑𝑚 for a given modality 𝑚 is computed as follows: [URL 🔗](#page-0)

where ×1 3 is the 𝑚𝑜𝑑𝑒 − (1 3) tensor contraction. The feature-aware

balanced representation

Z𝑓−𝑎𝑑𝑎𝑝𝑡 𝑚

∈ R𝑁×𝑑𝑚

of the conversation C

for a given modality

m is computed as:

√︁

√︁ [URL 🔗](#page-0)

where

⊙ denotes the element-wise product,

𝑑𝑘

is a scaling factor. G𝐾 and G𝑄 are

where

More specifically, the modality

𝑚

core tensor

computed using a Linear Transform (Figure 3 ), as expressed below: [URL 🔗](#page-0)

𝛽 ∈ [0, 1] is a balancing parameter to regulate the contribu-

tion of the original unimodal feature vector

Z𝑚.

## 3.4 Adaptive Modality Weighting (AMW)

Our key focus is to achieve balanced contributions from each modal- ity during the training. Similar to [40], we observe the imbalance problem in multimodal ERC through experiments analyzing the modality-wise weight in norm of each label during training. Appar- ently, the dominant unimodal encoder, e.g., text, tends to have its weight in norm increase much faster than the weaker modalities, [URL 🔗](#page-0)

where 𝑚 ∈ {𝑡, 𝑎, 𝑣}, 𝑊(1) ∈ R𝑑𝑚×𝑟𝑠 ,𝑊(2) ∈ R𝑑𝑚×𝑟𝑤 , 𝑊(1) ∈

𝑄𝑚

𝑄𝑚

R𝑑𝑚×𝑟𝑠 ,𝑊(2) ∈ R𝑑𝑚×𝑟𝑤 are the linear transformation matrix; ⊗1

𝐾𝑚

denotes the mode-1 Khatri-Rao product.

𝐾𝑚


*Figure 3: Linear Transform block to compute core tensor.*

i.e., audio and visual, leading to divergent unimodal logit scores and distorting the joint fusion representation. Inspired by [35, 45], we propose to incorporate modality-wise L2 normalization to prop- erly weight features, mitigating imbalances arising from differing data distributions and noise levels across modalities. This dynamic adjustment prevents any single modality from dominating the fu- sion process, thus enhancing overall performance. Therefore, the modality-level balanced representation Z𝑚−𝑎𝑑𝑎𝑝𝑡 of the given con- versation is calculated as follows: [URL 🔗](#page-0)

where 𝑊𝑚 ∈ R𝑑𝑚×| E| symbolizes the output matrix of the model pertaining to modality 𝑚, and E is the set of emotion classes.

For emotion recognition, we feed Z𝑚−𝑎𝑑𝑎𝑝𝑡 , into the mulilayer preceptron (MLP) with ReLU activation function to compute the output ˆ𝑦𝑖 ∈ R𝑁×| E| .

The output ˆ𝑦𝑖 is utilized to predict emotion labels.

## 3.5 Learning

First, we investigate the standard cross-entropy loss for this down- stream task, i.e., mutilmodal ERC as:

where 𝐵 is the batch size.

Second, in order to align between the original unimodal repre- sentation of modality 𝑚 and its respective feature-aware attention weights as Eq (6), we employ Attention Mapping Network as fol- lows: [URL 🔗](#page-0)

where Φ𝑚 (·) is a feed-forward neural network with the parameter 𝜓(𝑚) , 𝐴ˆ𝑡𝑡𝑚 ∈ R𝑁×𝑑𝑚 is the feature-aware self-attention weights of the modality 𝑚. To enhance feature-level balance across all modali- ties, we introduce a L1-norm loss L𝑓𝑒𝑎𝑡𝑢𝑟𝑒 as:

«

Additionally, we also consider the modality-level balance loss

L𝑚𝑜𝑑𝑎𝑙 , which is computed as:

¬

where Z𝑚−𝑎𝑑𝑎𝑝𝑡 represents the output of the 𝑗-th class for the 𝑖-th

𝑗

sample. Finally, we combine the all loss functions into a joint objec- tive function, which is used to optimize all trainable parameters in an end-to-end manner:

Recent studies have brought attention to the challenge of han- dling imbalanced optimization in joint learning models, particularly when dealing with multiple modalities. Peng et al. [26] introduce the OGM-GE method to address optimization imbalances encountered during the simultaneous training of dual-modal systems, i.e., visual and audio. However, directly applying the OGM-GE method to our framework is not practical as it only deals with two modalities. In contrast, our framework caters to more than two modalities across different domains, specifically tailored for the multimodal ERC task. Therefore, leanrable parameter ofencoder layer is optimized during training process as the following strategy: [URL 🔗](#page-0)

where ˆ𝑔(𝜃𝑚𝑡) = 𝑜 1 Í 𝑥 ∈ 𝐵𝑡∇𝜃𝑚 𝑡 𝓁 (𝑥, 𝜃 𝑡 (𝑖 ) ) represents an unbiased

estimation of the full gradient ∇𝜃𝑚𝓁 (𝑥, 𝜃

𝑡 (𝑖 ) ) using a random mini-

𝑡

batch 𝐵𝑡 chosen at the 𝑡-th step with size 𝑜. The term ∇𝜃𝑚𝓁 (𝑥, 𝜃 𝑡 (𝑖 ) )

𝑡

denotes the gradient with respect to 𝐵𝑡 .

We adjust the balance of modalities through gradient parameter adjustments. For each output at step 𝑡, we compute the discrepancy ratio for each modality using the softmax of the cosine similarity between the output weights and the corresponding feature vectors:

where

I𝑘=𝑦𝑗 equals 1 if 𝑘 = 𝑦𝑗 and 0 otherwise, and softmax(.)

estimates the unimodal performance of the multimodal model,

𝑀

denotes the count of modalities. Specifically, for the multimodal ERC task under consideration, we delineate three modalities: text (𝑡), audio (𝑎), and visual (𝑣). The discrepancy ratio is calculated as:

The learnable parameters are updated according to:

where the modulation coefficient 𝑘𝑚 is determined by 1 − tanh(𝛼 ·

𝑡

𝜌𝑚 𝑡 ) if 𝜌𝑚 > 1, and 1 otherwise. Here, 𝛼 is a hyperparameter

𝑡

controlling the degree of modulation. Additionally, to enhance

the adaptability of the modulation process, Gaussian noise

ℎ(𝜃 𝑡 (𝑖 ) )

N(0, Í𝑠𝑔𝑑 (𝜃 𝑡 (𝑖 ) )) is introduced after

sampled from a distribution parameter updates:

Training Optimization Strategy The training process ofAda2I

is illustrated in Algorithm 1. [URL 🔗](#page-0)


## Algorithm 1 Ada2I Training Procedure

Input: The training set D = { (𝑥𝑡 𝑖 , 𝑥𝑎 𝑖 , 𝑥𝑣 𝑖 ), 𝑦𝑖 }𝑁 𝑖=1, 𝑚 ∈ {𝑡, 𝑎, 𝑣} Output: Prediction emotion label ˆ𝑦

for each training epoch do

for minibatch B = { (𝑥𝑡 𝑖 , 𝑥𝑎 𝑖 , 𝑥𝑣 𝑖 ), 𝑦𝑖 }𝑁 𝑖=1 } sampled from D do #Refer to Subsection 3.2 [URL 🔗](#page-0)

Encode unimodal feature X𝑚 to Z𝑚 as Eq (2) [URL 🔗](#page-0)

#Refer to Subsection 3.3 [URL 🔗](#page-0)

Multimodal feature representation as Eq (3) [URL 🔗](#page-0)

Calculate coefficient matrix Θ𝑚 as Eq (4) [URL 🔗](#page-0)

Calculate modality-aware attention 𝐴𝑡𝑡𝑚 as Eq (6) [URL 🔗](#page-0)

Compute fused feature Z𝑓−𝑎𝑑𝑎𝑝𝑡

𝑚 with 𝛽 using Eq (7) [URL 🔗](#page-0)

#Refer to Subsection 3.4 [URL 🔗](#page-0)

Compute logit output Z𝑚−𝑎𝑑𝑎𝑝𝑡 with modality-wise L2 normal- [URL 🔗](#page-0)

ization as Eq (8) [URL 🔗](#page-0)

Produce prediction of multimodal data ˆ𝑦𝑖 as Eq (9) #Refer to Subsection 3.5 [URL 🔗](#page-0)

Use cross-entropy loss to calculate L𝑐𝑙𝑠 as Eq (10) [URL 🔗](#page-0)

Use 𝐿1 to calculate L𝑓𝑒𝑎𝑡𝑢𝑟𝑒 as Eq (12) [URL 🔗](#page-0)

Use cross-entropy to calculate L𝑚𝑜𝑑𝑎𝑙 as Eq (13) [URL 🔗](#page-0)

Add L𝑓𝑒𝑎𝑡𝑢𝑟𝑒 , L𝑚𝑜𝑑𝑎𝑙 and L𝑐𝑙𝑠 to compute L𝑚𝑎𝑖𝑛 as Eq (14) [URL 🔗](#page-0)

𝑠𝑚

𝑡

Compute discrepancy ratio 𝜌𝑚 =

𝑡

Compute modulation coefficient 𝑘𝑚

𝑡

end for

end for

min𝑚∈{𝑡,𝑎,𝑣} (𝑠 𝑡 𝑗 )

## 3.6 Datasets

Datasets: We consider three benchmark datasets for multimodal ERC namely: IEMOCAP [3], MELD [28], and CMU-MOSEI [1]. The dataset statistics are illustrated in Table 1. [URL 🔗](#page-0)

*Table 1: Data Statistics*

| Datasets | Dialogues |   | Utterances |   |
| --- | --- | --- | --- | --- |
|   |   | train valid test |   | train valid test |
| IEMOCAP | 120 | 31 | 5,810 | 1,623 |
| MELD |   |   |   | 1,039 114 280 9,989 1,109 2,610 |
|   |   |   |   | CMU-MOSEI 2,248 300 676 16,326 1,871 4,659 |

IEMOCAP. This dataset comprises 12 hours of video recordings of dyadic conversations involving 10 speakers. It includes 151 dia- logues, segmented into 7,433 utterances, each annotated with one of six emotion labels: happy, sad, neutral, angry, excited, or frustrated.

MELD. This dataset is based on the TV series Friends, includes 13,709 video clips featuring multi-party conversations, each labeled with one ofEkman’s six universal emotions: joy, sadness, fear, anger, surprise, and disgust.

CMU-MOSEI:. This dataset is a prominent resource for sentiment and emotion analysis, comprises 3,228 YouTube videos divided into 23,453 segments, featuring contributions from 1,000 speakers covering 250 topics. It includes six emotion categories: happy, sad, angry, scared, disgusted, and surprised, with sentiment intensity ranging from -3 to 3.

*Table 2: Hyper-parameter settings*

| Parameter/Module | IEMOCAP MELD CMU-MOSEI |
| --- | --- |
| Text Feature Extraction | sBERT1 |
|   | Audio Feature Extraction Wave2vec-Large [29], OpenSmile [5] |
|   | Visual Feature Extraction MTCNN [42], MA-Net2, DenseNet [12] |
| Text embedding dim. 𝑑𝑡 | 768 768 768 |
| Audio embedding dim. 𝑑𝑎 | 512 300 512 |
| Visual embedding dim. 𝑑𝑣 | 1024 342 1024 |
| hidden dim | 300 200 500 |
| tensor rank | 11 6 10 |
| 𝜂 | 0.037 0.4 0.4 |
| 𝛽 | 0.01 0.55 0.2 |
| learning rate | 1.7e-4 1.2e-4 1.9e-4 |
| batch size | 10 10 32 |
| epoch | 50 50 30 |

## 4 EXPERIMENTIAL SETUP

## 4.1 Baselines and Evaluation Metrics

Baselines: Ada2I is compared against several state-of-the-art (SOTA) baseline approaches for evaluating performance in multi- modal ERC, particularly addressing modality imbalance problems. For the IEMOCAP and MELD datasets, we consider baseline models such as DialogueRNN [23], DialogueGCN [8], MMGCN [38], BiD- DIN [41], and MM-DFN [11]. We report the best results obtained from [37], which enhanced these models to address modality imbal- ance. Additionally, we consider other SOTA models for multimodal ERC that do not explicitly address modality imbalance, including COGMEN [16], CORECT [24], GraphMFT [18], DF-ERC [17], and AdaIGN [33]. [URL 🔗](#page-0)

For the CMU-MOSEI dataset, we evaluated various baseline mod- els for sentiment classification tasks, which include both 2-class sen- timent, featuring only positive and negative sentiment, and 7-class sentiment, ranging from highly negative (-3) to highly positive (+3). These baseline models include Multilouge-Net [31], TBJE [4], COG- MEN [16], CORECT [24], OGM-GE [26], and I2MCL [46]. Notably, OGM-GE and I2MCL specifically address the issue of imbalanced modalities in multimodal ERC, whereas the others do not. [URL 🔗](#page-0)

Evaluation Metrics: Similar to prior studies [23, 37, 38], we eval- uate the effectiveness of emotion recognition using Accuracy (Acc) and Weighted F1 Score (W-F1) as our primary evalucation metrics. [URL 🔗](#page-0)

## 4.2 Experimental Settings

We derive multimodal features for each utterance from acoustic, lexical, and visual modalities using a combination of models and pre-trained models, as outlined in Table 2. [URL 🔗](#page-0)

We employ PyTorch3 for training our architecture and Comet4 for logging all experiments, leveraging its Bayesian optimizer for hyperparameter tuning. Additional parameters can be found in Table 2. [URL 🔗](#page-0)


*Table 3: Comparison of results in the multimodal setting of Ada2I with the modality-balanced baseline model enhanced by FAGM [37] (denoted by †). The best performance is indicated in bold, and the second-best performance is underlined. [URL 🔗](#page-0)*

|   |   |   | IEMOCAP |   |   | MELD |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methods | T+A+V | T+A | T+V | A+V | T+A+V | T+A | T+V | A+V |
|   | W-F1 Acc W-F1 Acc W-F1 Acc W-F1 Acc W-F1 Acc W-F1 Acc W-F1 Acc W-F1 Acc |   |   |   |   |   |   |   |
|   |   | DialogueRNN† 61.31 61.61 61.90 61.98 60.19 59.95 48.31 50.71 |   |   | 56.42 58.05 56.46 58.01 55.67 57.39 40.46 45.39 |   |   |   |
| DialogueGCN† 62.76 63.22 64.36 |   | 64.39 | 61.25 62.23 | 49.20 49.85 | 54.61 58.96 54.80 57.28 55.26 57.10 10.02 44.44 |   |   |   |
| BiDDIN† | 58.81 58.84 58.88 58.16 59.04 58.96 46.36 46.77 |   |   |   | 57.47 59.18 56.56 58.05 56.93 58.10 44.39 |   |   | 48.62 |
| MM-DFN† | 64.92 64.57 |   |   | 63.91 64.20 61.02 60.60 54.48 55.03 | 55.75 60.8 57.10 60.00 57.73 |   | 60.65 | 42.05 48.66 |
| MMGCN† | 64.53 64.51 63.25 63.40 61.02 61.06 54.14 54.90 |   |   |   | 58.48 61.15 | 57.59 60.69 | 57.14 59.46 43.49 48.43 |   |
|   |   |   |   | Ada2I (Ours) 68.97 68.76 66.91 67.28 65.48 65.43 55.16 55.64 60.38 63.03 60.08 62.64 58.62 61.95 55.16 55.64 |   |   |   |   |
| Δ(%) | ↑4.05 ↑4.19 ↑2.55 ↑2.89 ↑4.23 ↑3.20 ↑0.68 ↑0.61 ↑1.90 ↑1.88 ↑2.49 ↑1.95 ↑0.89 ↑1.30 ↑10.77 ↑6.98 |   |   |   |   |   |   |   |

## 5 RESULTS AND DISCUSSION

## 5.1 Performance Comparison against Baselines

IEMOCAP and MELD dataset: As depicted in Table 3, our model Ada2I performs better than the previous SOTA baselines in the con- text of balanced modality consideration on all modality combina- tions on both datasets. Indeed, in the AV modality pair on the MELD dataset, traditionally deemed the weakest, we observe a substan- tial performance boost in Multimodal ERC. Specifically, there is a noteworthy enhancement of10.77% on WF1 and 6.98% on Accuracy compared to the previous SOTA model. This progress effectively reduces the performance discrepancy compared to modality pairs where text plays a dominant role. [URL 🔗](#page-0)

We also compare Ada2I with SOTA baseline models for multi- modal ERC, particularly those focusing solely on multimodal fusion and architectural design without addressing modality imbalance. Figure 4b demonstrates that our proposed Ada2I significantly re- duces the performance gap in WF1 between learning from all three modalities simultaneously (T+A+V) and pair-wise modality com- binations on the MELD dataset. Most notably, with the weaker modality pair (audio+visual) consistently lagging behind in per- formance compared to the full modality combination (i.e., with AdaIGN, this gap is 23.12%), Ada2I boosts the model and short- ens the gap to only 5.22%. Similarly, with the text+audio (T+A) and text+visual (T+V) pairs, this gap is also substantially reduced, indicating that the model has learned in a more balanced man- ner, leveraging additional useful information from non-dominant modalities. The significant improvement is similarly observed on the IEMOCAP dataset in Figure 4a. [URL 🔗](#page-0)

CMU-MOSEI dataset: Table 4 shows that Ada2I outperforms all baseline models. Specifically, when compared to OGM-GE and I2MCL, two models proposed for addressing modality imbalance during training, Ada2I demonstrates superior performance across all modality combinations. When compared to other baseline mod- els that do not consider modality balancing, Ada2I also demon- strates significant balancing capabilities, reducing the performance gap between modality pairs. For instance, in the CORECT model, the gap between T+A+V and A+V is 15.09% for 2-class sentiment, and this figure increases to 21.76% for 7-class sentiment. However, with Ada2I, these gaps are significantly reduced to 10.32% and 13.07%, respectively, underscoring the effectiveness of Ada2I in addressing modality imbalances. [URL 🔗](#page-0)

*Figure 4: Performance gap visualizations between the multi- modal setting (T+A+V) and pair-wise modality combinations are evaluated using the W-F1 metric across the IEMOCAP and MELD datasets.*

## 5.2 Ablation Study

5.2.1 Balancing Interpretation. We conduct ablation studies with the two main modules of the model, AMW and AFW, to assess their impact on the Ada2I model. Additionally, through the Discrepancy Ratio, we interpret the model’s balancing by observing its changes. A smaller Discrepancy Ratio indicates a more balanced optimization process. Figure 5 shows that the discrepancy ratios 𝜌𝑡 , 𝜌𝑣, and 𝜌𝑎 significantly decrease when both AMW and AFW are combined within Ada2I, with all ratios approaching approximately 1 on the IEMOCAP dataset. In contrast, when one of the modules is ablated, the ratios for audio (𝜌𝑎) and visual (𝜌𝑣) are approximately 1.5, while for text, it increases to around 3. Similarly, on the MELD dataset, our proposed model Ada2I has reduced this discrepancy [URL 🔗](#page-0)


*Table 4: Results on the CMU-MOSEI dataset with accuracy (Acc.) as the metric. The best performance is in bold. Cells with “-” indicate missing results, and † denotes results repro- duced from the code provided in the original paper.*

| Methods | 2-class |   |   | 7-class |   |   |
| --- | --- | --- | --- | --- | --- | --- |
|   | T+A+V T+A T+V A+V T+A+V +TA T+V A+V |   |   |   |   |   |
| Multilouge-Net [31] | 82.10 80.18 80.06 | 75.16 | 44.83 | - | - | - |
| TBJE [4] | 81.50 82.40 - | - | 44.40 45.50 |   | - | - |
| COGMEN† [16] | 82.95 85.00 | 82.99 65.95 | 43.90 44.31 42.68 24.27 |   |   |   |
| CORECT† [24] | 83.98 84.28 82.83 68.89 |   | 46.31 |   | 44.89 43.76 24.55 |   |
| I2MCL [46] | 81.05 - - | - | - | - | - | - |
| OGM-GE† [26] | 84.58 84.03 83.67 | 71.53 | 45.43 43.68 44.44 |   |   | 31.53 |
| Ada2I (Ours) | 85.25 85.08 85.21 | 74.93 | 47.71 47.35 47.37 34.64 |   |   |   |
| Δ(%) | ↑0.67 | ↑0.08 ↑1.54 ↓0.23 | ↑2.28 |   | ↑1.85 ↑2.93 ↑3.11 |   |

*Figure 5: The change of the discrepancy ratio 𝜌𝑡, 𝜌𝑎, 𝜌𝑣 on the IEMOCAP and MELD datasets during training, along with various ablation tests including without AMW and without AFW, are compared to the Ada2I model.*

ratio of text from over 4 (w/o AFW) to approximately half, reaching around 2, while for audio and visual, it brings them close to the 1 mark. In summary, the combined design ofboth modules AMW and AFW enhances balanced learning across modalities during training, highlighting the significance and inseparability of feature-level and modality-level balancing.

5.2.2 Effect ofWeight Normalization. As mentioned earlier, the unimodal weights also directly influence the encoder updating process. The imbalanced weight components induce gradients and subsequently lead to the inconsistent convergence of unimodalities. Here, we provide a clearer visualization of these unimodal weights before imbalance processing (Only Encoder) and in the Ada2I model in Figure 6 for the IEMOCAP dataset. It is evident that with Only Encoder, the text encoder (dominant modality) weight in norm grows much faster than audio and visual. After balancing, our model exhibits a more balanced optimization process. [URL 🔗](#page-0)

5.2.3 EffectofModule. Table 5 provides an ablation on the modules. AFW and AMW are two closely linked and crucial modules in [URL 🔗](#page-0)

Epoch

Epoch

*Figure 6: Modality-wise weights of each label normalized for the IEMOCAP dataset*

Ada2I, ensuring model stability. Furthermore, Ada2I with training optimization balances the training across three modalities (text, audio, visual), preventing the text modality from dominating the others.

*Table 5: Ablation studies of Ada2I on AFW, AMW, and train- ing strategy. The symbol ↓ denotes the reduction in perfor- mance of the variants compared to Ada2I.*

| Modules | IEMOCAP |   | MELD |
| --- | --- | --- | --- |
| W-F1 | Acc | W-F1 | Acc |
| w/o AFW 66.24(↓2.73) 65.99(↓2.77) 59.65(↓0.73) 62.45(↓0.58) |   |   |   |
| w/o AMW 66.11(↓2.86) 65.87(↓2.89) 58.87(↓1.51) 61.13(↓1.90) |   |   |   |
|   |   |   | w/o traning optimization 67.95(↓1.02) 68.08(↓0.68) 58.13(↓2.25) 59.92(↓3.11) |
| Ada2I (Ours) 68.97 | 68.76 | 60.38 | 63.03 |

## 6 CONCLUSION

In this work, we present Ada2I, a framework designed to address modality imbalances and optimize learning in multimodal ERC. We identify and analyze existing issues in current ERC models that overlook the imbalance problem. From there, we propose a solution comprising integral modules: Adaptive Feature Weighting (AFW) and Adaptive Modality Weighting (AMW). The former enhances intra-modal representations for feature-level balancing, while the latter optimizes inter-modal learning weights with the balancing at modality level. Furthermore, we introduce a refined disparity ratio to optimize training, offering a straightforward yet effective measure to evaluate the model’s overall discrepancy when handling multiple modalities simultaneously. Extensive experiments on the IEMOCAP, MELD, and CMU-MOSEI datasets validate its effective- ness, showcasing SOTA performance. In the future, we anticipate enhancing the efficiency of the framework and maximizing the utilization of emotional cues.

## ACKNOWLEDGMENTS

Cam-Van Thi Nguyen was funded by the Master, PhD Scholar- ship Programme of Vingroup Innovation Foundation (VINIF), code VINIF.2023.TS147.


## REFERENCES

- [1] AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis- Philippe Morency. 2018. Multimodal Language Analysis in theWild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Iryna Gurevych and Yusuke Miyao (Eds.). Association for Computational Linguistics, Melbourne, Australia, 2236–2246. https://doi.org/10.18653/v1/P18- 1208 [URL 🔗](https://doi.org/10.18653/v1/P18-1208)

- [2] Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. 2018. Multi- modal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence 41, 2 (2018), 423–443.

- [3] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. 2008. IEMOCAP: Interactive emotional dyadic motion capture database. Language resources and evaluation 42, 4 (2008), 335–359.

- [4] Jean-Benoit Delbrouck, Noé Tits, Mathilde Brousmiche, and Stéphane Dupont. 2020. A Transformer-based joint-encoding for Emotion Recognition and Senti- ment Analysis. In Second Grand-Challenge andWorkshop on Multimodal Language (Challenge-HML), Amir Zadeh, Louis-Philippe Morency, Paul Pu Liang, and Sou- janya Poria (Eds.). Association for Computational Linguistics, Seattle, USA, 1–7. https://doi.org/10.18653/v1/2020.challengehml-1.1 [URL 🔗](https://doi.org/10.18653/v1/2020.challengehml-1.1)

- [5] Florian Eyben, Martin Wöllmer, and Björn Schuller. 2010. Opensmile: the munich versatile and fast open-source audio feature extractor. In Proceedings ofthe 18th ACM international conference on Multimedia. 1459–1462. [URL 🔗](https://doi.org/10.18653/v1/2020.challengehml-1.1)

- [6] Yunfeng Fan, Wenchao Xu, Haozhao Wang, Junxiao Wang, and Song Guo. 2023. PMR: Prototypical Modal Rebalance for Multimodal Learning. In ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20029– 20038. Proceedings

- [7] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. 2019. Slow- fast networks for video recognition. In Proceedings ofthe IEEE/CVF international conference on computer vision. 6202–6211.

- [8] Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya, and Alexander Gelbukh. 2019. DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 154–164. https://doi.org/10.18653/v1/D19-1015 [URL 🔗](https://doi.org/10.18653/v1/D19-1015)

- [9] Devamanyu Hazarika, Soujanya Poria, Rada Mihalcea, Erik Cambria, and Roger Zimmermann. 2018. ICON: Interactive Conversational Memory Network for Multimodal Emotion Detection. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Ellen Riloff, David Chiang, Julia Hock- enmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, Brussels, Belgium, 2594–2604. https://doi.org/10.18653/v1/D18-1280 [URL 🔗](https://doi.org/10.18653/v1/D18-1280)

- [10] Devamanyu Hazarika, Soujanya Poria, Amir Zadeh, Erik Cambria, Louis-Philippe Morency, and Roger Zimmermann. 2018. Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Videos. In Proceedings ofthe 2018 Con- ference ofthe North American Chapter ofthe Association for Computational Lin- guistics: Human Language Technologies, Volume 1 (Long Papers), Marilyn Walker, Heng Ji, and Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans, Louisiana, 2122–2132. https://doi.org/10.18653/v1/N18-1193 [URL 🔗](https://doi.org/10.18653/v1/D18-1280)

- [11] Dou Hu, Xiaolong Hou, LingweiWei, Lianxin Jiang, and Yang Mo. 2022. MM-DFN: Multimodal dynamic fusion network for emotion recognition in conversations. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 7037–7041. [URL 🔗](https://doi.org/10.18653/v1/N18-1193)

- [12] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. 2017. Densely Connected Convolutional Networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

- [13] Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, and Longbo Huang. 2021. What makes multi-modal learning better than single (provably). Advances in Neural Information Processing Systems 34 (2021), 10944–10956.

- [14] Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. 2022. Modality competition: What makes joint training of multi-modal network fail in deep learning?(provably). In International Conference on Machine Learning. PMLR, 9226–9259.

- [15] Adrián Javaloy, Maryam Meghdadi, and Isabel Valera. 2022. Mitigating modality collapse in multimodal VAEs via impartial optimization. In International Confer- ence on Machine Learning. PMLR, 9938–9964.

- [16] Abhinav Joshi, Ashwani Bhat, Ayush Jain, Atin Singh, and Ashutosh Modi. 2022. COGMEN: COntextualized GNN based multimodal emotion recognitioN. In Proceedings ofthe 2022 Conference ofthe North American Chapterofthe Association for Computational Linguistics: Human Language Technologies. 4148–4164.

- [17] Bobo Li, Hao Fei, Lizi Liao, Yu Zhao, Chong Teng, Tat-Seng Chua, Donghong Ji, and Fei Li. 2023. Revisiting disentanglement and fusion on modality and context in conversational multimodal emotion recognition. In Proceedings ofthe 31st ACM International Conference on Multimedia. 5923–5934.

- [18] Jiang Li, Xiaoping Wang, Guoqing Lv, and Zhigang Zeng. 2023. GraphMFT: A graph network based multimodal fusion technique for emotion recognition in conversation. Neurocomputing 550 (2023), 126427.

- [19] Zheng Lian, Bin Liu, and Jianhua Tao. 2021. CTNet: Conversational transformer network for emotion recognition. IEEE/ACMTransactions on Audio, Speech, and Language Processing 29 (2021), 985–1000.

- [20] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Yu- fan Chen, PeterWu, Michelle A Lee, Yuke Zhu, et al. 2021. MultiBench: Multiscale Benchmarks for Multimodal Representation Learning. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

- [21] Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. 2022. Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions. arXiv preprint arXiv:2209.03430 (2022).

- [22] Xun Lin, Shuai Wang, Rizhao Cai, Yizhong Liu, Ying Fu, Zitong Yu, Wenzhong Tang, and Alex Kot. 2024. Suppress and Rebalance: Towards Generalized Multi- Modal Face Anti-Spoofing. arXiv preprint arXiv:2402.19298 (2024).

- [23] Navonil Majumder, Soujanya Poria, Devamanyu Hazarika, Rada Mihalcea, Alexander Gelbukh, and Erik Cambria. 2019. Dialoguernn: An attentive rnn for emotion detection in conversations. In artificial intelligence, Vol. 33. 6818–6825. Proceedings ofthe AAAIconference on

- [24] Cam-Van Thi Nguyen, Tuan Mai, Son The, Dang Kieu, and Duc-Trong Le. 2023. Conversation Understanding using Relational Temporal Graph Neural Networks with Auxiliary Cross-Modality Interaction. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 15154–15167. https://doi.org/10.18653/v1/2023.emnlp-main.937

- [25] Cam-Van Thi Nguyen, Cao-Bach Nguyen, Duc-Trong Le, and Quang-Thuy Ha. 2024. Curriculum Learning Meets Directed Acyclic Graph for Multimodal Emo- tion Recognition. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakri- ani Sakti, and Nianwen Xue (Eds.). ELRA and ICCL, Torino, Italia, 4259–4265. https://aclanthology.org/2024.lrec-main.380 [URL 🔗](https://doi.org/10.18653/v1/2023.emnlp-main.937)

- [26] Xiaokang Peng, Yake Wei, Andong Deng, DongWang, and Di Hu. 2022. Balanced multimodal learning via on-the-fly gradient modulation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8238–8247. [URL 🔗](https://aclanthology.org/2024.lrec-main.380)

- [27] Soujanya Poria, Erik Cambria, Devamanyu Hazarika, Navonil Majumder, Amir Zadeh, and Louis-Philippe Morency. 2017. Context-Dependent Sentiment Analy- sis in User-Generated Videos. In Proceedings ofthe 55th Annual Meeting ofthe Association forComputational Linguistics (Volume 1: Long Papers), Regina Barzilay and Min-Yen Kan (Eds.). Association for Computational Linguistics, Vancouver, Canada, 873–883. https://doi.org/10.18653/v1/P17-1081

- [28] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, Anna Korhonen, David Traum, and Lluís Màrquez (Eds.). Association for Computational Linguistics, Florence, Italy, 527–536. https://doi.org/10.18653/v1/P19-1050 [URL 🔗](https://doi.org/10.18653/v1/P17-1081)

- [29] Steffen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. 2019. wav2vec: Unsupervised pre-training for speech recognition. arXiv:1904.05862 (2019). arXiv preprint [URL 🔗](https://doi.org/10.18653/v1/P19-1050)

- [30] Weizhou Shen, Siyue Wu, Yunyi Yang, and Xiaojun Quan. 2021. Directed Acyclic Graph Network for Conversational Emotion Recognition. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (Eds.). Association for Computational Linguistics, Online, 1551–1560. https://doi.org/ 10.18653/v1/2021.acl-long.123

- [31] Aman Shenoy and Ashish Sardana. 2020. Multilogue-Net: A Context-Aware RNN for Multi-modal Emotion Detection and Sentiment Analysis in Conversation. In Second Grand-Challenge andWorkshop on Multimodal Language (Challenge- HML), Amir Zadeh, Louis-Philippe Morency, Paul Pu Liang, and Soujanya Poria (Eds.). Association for Computational Linguistics, Seattle, USA, 19–28. https: //doi.org/10.18653/v1/2020.challengehml-1.3 [URL 🔗](https://doi.org/10.18653/v1/2021.acl-long.123)

- [32] Jiajia Tang, Kang Li, Ming Hou, Xuanyu Jin, Wanzeng Kong, Yu Ding, and Qibin Zhao. 2022. MMT: Multi-Way Multi-Modal Transformer for Multimodal Learn- ing. In Proceedings ofthe Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, LD Raedt, Ed. International Joint Conferences on Artificial Intelligence Organization, Vol. 7. 3458–3465. [URL 🔗](https://doi.org/10.18653/v1/2020.challengehml-1.3)

- [33] Geng Tu, Tian Xie, Bin Liang, Hongpeng Wang, and Ruifeng Xu. 2024. Adaptive Graph Learning for Multimodal Conversational Emotion Detection. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 38. 19089–19097.

- [34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems. 6000–6010.


- [35] Feng Wang, Xiang Xiang, Jian Cheng, and Alan Loddon Yuille. 2017. Normface: L2 hypersphere embedding for face verification. In Proceedings ofthe 25th ACM international conference on Multimedia. 1041–1049.

- [36] Weiyao Wang, Du Tran, and Matt Feiszli. 2020. What makes training multi- modal classification networks hard?. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 12695–12705.

- [37] Yunxiao Wang, Meng Liu, Zhe Li, Yupeng Hu, Xin Luo, and Liqiang Nie. 2023. Unlocking the Power of Multimodal Learning for Emotion Recognition in Con- versation. In Proceedings ofthe 31st ACM International Conference on Multimedia. 5947–5955.

- [38] Yinwei Wei, XiangWang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. 2019. MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video. In Proceedings ofthe 27th ACM international conference on multimedia. 1437–1445.

- [39] Nan Wu, Stanislaw Jastrzebski, Kyunghyun Cho, and Krzysztof J Geras. 2022. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In International Conference on Machine Learning. PMLR, 24043–24055.

- [40] Ruize Xu, Ruoxuan Feng, Shi-Xiong Zhang, and Di Hu. 2023. MMCosine: Multi- Modal Cosine Loss Towards Balanced Audio-Visual Fine-Grained Learning. In

- ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 1–5.

- [41] Dong Zhang, Weisheng Zhang, Shoushan Li, Qiaoming Zhu, and Guodong Zhou. 2020. Modeling both intra-and inter-modal influence for real-time emotion de- tection in conversations. In Proceedings ofthe 28th ACM International Conference on Multimedia. 503–511.

- [42] Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Qiao Yu. 2016. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters 23, 10 (2016), 1499–1503.

- [43] Qingyang Zhang, Yake Wei, Zongbo Han, Huazhu Fu, Xi Peng, Cheng Deng, Qinghua Hu, Cai Xu, JieWen, Di Hu, et al. 2024. Multimodal fusion on low-quality data: A comprehensive survey. arXiv preprint arXiv:2404.18947 (2024).

- [44] Qibin Zhao, Guoxu Zhou, Shengli Xie, Liqing Zhang, and Andrzej Cichocki. 2016. Tensor ring decomposition. arXiv preprint arXiv:1606.05535 (2016).

- [45] Yutong Zheng, Dipan K Pal, and Marios Savvides. 2018. Ring loss: Convex feature normalization for face recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 5089–5097.

- [46] Yuwei Zhou, Xin Wang, Hong Chen, Xuguang Duan, and Wenwu Zhu. 2023. Intra-and Inter-Modal Curriculum for Multimodal Learning. In Proceedings ofthe 31st ACM International Conference on Multimedia. 3724–3735.
