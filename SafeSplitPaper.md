SafeSplit: A Novel Defense Against Client-Side
Backdoor Attacks in Split Learning (Full Version)*
Phillip Rieger
Technical University of Darmstadt
Alessandro Pegoraro
Technical University of Darmstadt
Kavita Kumari
Technical University of Darmstadt
Tigist Abera
Technical University of Darmstadt
Jonathan Knauer
Technical University of Darmstadt
Ahmad-Reza Sadeghi
Technical University of Darmstadt
Abstract—Split Learning (SL) is a distributed deep learning
approach enabling multiple clients and a server to collaboratively
train and infer on a shared deep neural network (DNN) without
requiring clients to share their private local data. The DNN is
partitioned in SL, with most layers residing on the server and a
few initial layers and inputs on the client side. This configuration
allows resource-constrained clients to participate in training and
inference. However, the distributed architecture exposes SL to
backdoor attacks, where malicious clients can manipulate local
datasets to alter the DNN’s behavior. Existing defenses from
other distributed frameworks like Federated Learning are not
applicable, and there is a lack of effective backdoor defenses
specifically designed for SL.
We present SafeSplit, the first defense against client-side
backdoor attacks in Split Learning (SL). SafeSplit enables the
server to detect and filter out malicious client behavior by
employing circular backward analysis after a client’s training
is completed, iteratively reverting to a trained checkpoint where
the model under examination is found to be benign. It uses a
two-fold analysis to identify client-induced changes and detect
poisoned models. First, a static analysis in the frequency domain
measures the differences in the layer’s parameters at the server.
Second, a dynamic analysis introduces a novel rotational distance
metric that assesses the orientation shifts of the server’s layer parameters
during training. Our comprehensive evaluation across
various data distributions, client counts, and attack scenarios
demonstrates the high efficacy of this dual analysis in mitigating
backdoor attacks while preserving model utility.
I. INTRODUCTION
Recently, deep neural networks (DNNs) have made significant
advances, leading to the development of new training frameworks
such as Large Language Models (LLMs)1, AI-based
image generation, and image recognition for self-driving cars.
Concurrently, the complexity of deployed DNNs has grown
*Please cite the version of this paper published at NDSS 2025 [42].
1Although transformers and diffusion models differ significantly from
traditional DNNs, they still consist of layers and trainable parameters, making
them suitable for distributed learning.
rapidly to manage the ever-increasing tasks, demanding more
robust computational resources. This increasing complexity
presents a substantial challenge for deploying such advanced
DNNs on resource-constrained devices without compromising
the privacy of potentially sensitive input data.
Split Learning (SL) is a class of distributed learning that
promises to reduce the computational load on the client
side without requiring the clients to share their data. In this
paradigm, the DNN’s architecture is split between client and
server, with the computationally intensive layers outsourced
to a server [53], [20]. Thus, it is a resource-friendly collaboration
between the clients and the server, unlike federated
learning [30], [18], [23]. The strengths of Split Learning have
been demonstrated on real-world data in medical contexts [63].
Further potential applications include financial services and
AI-based consumer services such as image editing on mobile
devices, analogously to Federated Learning that is widely
deployed as part of GBoard [34]. In the past, different
configurations for SL were developed, with the two most
prevalent being the vanilla and U-shaped configurations [18],
[50], [24], [6], [67]. In the vanilla configuration [50], [68],
the DNN is split into two segments at a specific ”cut layer.”
The smaller portion of the network resides on the client
side, while the larger part is on the server side. In the Ushaped
configuration [68], [18], the DNN is split into three
segments. The head segment, up to the first ”cut layer”, and
the tail segment, starting from the second ”cut layer”, reside
on the client side. The middle segment called the backbone,
is outsourced to the server because it is usually composed
of most of the layers and, therefore, is computationally more
intensive to train.
Training in SL proceeds sequentially, with clients queuing
for sessions. In the vanilla setup, each client trains its part of
the network up to the cut layer, transmitting outputs to the
server, which completes the training. In U-shaped systems,
the client model evaluates data up to the first cut layer during
forward propagation before sending the feature vectors to the
server. The server processes these vectors and returns the
output for the client to complete using the tail. During backpropagation,
the client calculates the loss and sends gradients
to the server, which computes the backbone’s gradients and
Network and Distributed System Security (NDSS) Symposium 2025
24-28 February 2025, San Diego, CA, USA
ISBN 979-8-9894372-8-3
https://dx.doi.org/10.14722/ndss.2025.241698
www.ndss-symposium.org
arXiv:2501.06650v2 [cs.CR] 24 Feb 2025
sends cut layer gradients back to the client. This setup allows
resource-constrained devices to train large models securely and
privately [53], [47], complying with data privacy standards like
GDPR [2] and HIPAA [1]. However, this setup also increases
the attack surface and results in a stronger threat model, since
the adversary can arbitrarily manipulate the loss. Thus, the rest
of the paper focuses on the U-shape configuration.
Attacks on SL. Recent works demonstrated the vulnerability
of SL to multiple attacks, including data reconstruction
attacks [33], label inference attacks [38], [15], and backdoor
attacks [6], [22], [64]. An adversary can launch a backdoor
attack on the server-side [13], [65] or the client-side [64], [22].
This paper focuses on client-side backdoor attacks, as clients
are more susceptible to attacks than well-protected servers.
Additionally, malicious clients in SL have an advantage due
to their access to data and labels, making it crucial for the
server to defend against such attacks to protect benign clients.
However, an efficient defense in SL is challenging because
each client uses the trained model of its predecessor as a
base, resulting in different starting models for each client. If a
previous client was malicious, subsequent benign clients may
unknowingly train on a poisoned model, compromising also
their results.
Although in the past various defenses were proposed to
mitigate attacks in other distributed learning paradigms, such
as Federated Learning [14], [17], [8], [5], [41], [11], they are
not applicable in SL due to the aforementioned sequential
training structure.
Our goal and contributions. To address the challenge of
client-side backdoor attacks in SL, we introduce SafeSplit, a
versatile and, to the best of our knowledge, the first backdoor
defense for the U-shaped SL paradigm deployed at the server.
SafeSplit operates on a rollback mechanism to employ circular
backward analysis after a client’s training ends, reverting to a
trained state where the client under examination emerges as
benign. That is, if the trained backbone at the server shows
backdoor characteristics, this rollback mechanism reverts to a
previous client’s trained state and continues the examination
to ensure a benign trained state is selected, bypassing all the
malicious client’s trained updates.
To identify poisoned training contributions and detect malicious
behavior, SafeSplit employs a two-fold strategy to perform
the static and dynamic analysis of the server’s backbone
during the rollback mechanism. These metrics are based on the
rationale that benign behavior contradicts the mispredictions
caused by backdoor behavior. Therefore, significant changes
need to be applied by the attacker to the model to introduce
new behavior into the model. First, the motivation to conduct
the static analysis of the server’s backbone is to measure
static characteristics of the backbone’s parameters, such as
the changes in the frequency domain, without considering
changes over time during training. An analogy for this strategy
would be analyzing a music recording to see how often certain
notes (frequencies) appear without considering how the music
changes over time. Here, we are only interested in the presence
of the frequency of notes, not in the sequence or evolution
of the music. Second, the motivation to conduct the dynamic
analysis is to assess how the orientation or configuration of
the backbone’s parameters changes throughout training. We
introduce a novel dynamic rotational distance metric that
measures the extent of dynamic shifts in the backbone’s
values or configurations, providing insights into how the values
evolve during training. Thus, the rotational distance metric
analyzes dynamic aspects, such as the flow and transition of
the backbone values. Using the music analogy, the metric
measures the transitions between the notes rather than their
bare presence. Therefore, SafeSplit uses both perspectives to
analyze the clients’ models and detect backdoors. Using this
analysis framework, we create a first novel robust defense
against the backdoor attacks in SL.
Two important things to note in the design of SafeSplit:
Firstly, we do not permanently remove malicious clients from
consideration. Instead, we skip the models of these clients, allowing
them to be reconsidered in subsequent training epochs
to ensure that misclassified benign clients are not unjustly
removed. Secondly, we deploy SafeSplit before a client starts
its training to select a benign starting point and ensure that
poisoned training contributions are effectively mitigated and
not used to train benign models. In summary, our contributions
include:
• We propose SafeSplit, the first defense framework designed
to mitigate backdoor attacks in Split Learning
(SL). SafeSplit accurately detects backdoor attacks and
reduces their impact while minimizing harmful effects
on the models’ utility. We conduct static and dynamic
analyses to inspect nuances in the applied model updates
(Sect. V-A).
• Our approach addresses the challenges introduced by SL’s
sequential structure through a circular defense mechanism
applied after each client’s training process. It enables the
early detection of malicious client behavior and mitigates
its impact before it influences the training of other benign
clients (Sect. V-B).
• We introduce a novel rotational distance metric that
measures, based on that angular displacement, how the
orientation or configuration of the backbone’s value at
the server changes throughout training (Sect. V-D). This
metric also captures dynamic nuances such as the orientation
or rotation of the changes that provide additional
information about learning objectives during the training
(Sect. V-D).
• We perform a deep analysis of the static changes introduced
through the local training by analyzing the model
updates in the frequency domain (Sect. V-C).
• To evaluate the effectiveness of our defense, we developed
various backdoor attacks, including poisoning and
semantic backdoor attacks. These attacks were applied
to datasets such as CIFAR-10, FMNIST, MNIST,CIFAR-
100, and GTSRB across different numbers of clients, attack
settings, and data scenarios. Our extensive evaluation
demonstrates the effectiveness of SafeSplit even against
2
Input
𝐹𝐹
(a) Connected DNN
𝐵𝐵
Input
𝐻𝐻 𝑇𝑇
(b) DNN split into head, backbone
and tail.
Fig. 1: Comparison of splitting the Deep Neural Network
(DNN) F into head (H), backbone (B), and tail (T), such
that F ≡ H ◦ B ◦ T. The head and tail are located on the
client side, and the backbone is on the server side.
defense-adapted attacks (Sect. VI).
With SafeSplit, we make the first step towards solving an open
challenge of mitigating client-side backdoor attacks on SL,
significantly reducing the backdoor impact while maintaining
the accuracy of the resulting model. We hope that future
research continues building on top of our scheme.
II. BACKGROUND
A. Split Learning
In the U-shaped Split Learning paradigm, N different
participants C1, . . . ,CN jointly train a DNN coordinated by a
central server. As shown in Figure 1, the typical architecture
of a connected DNN (Fig. 1a) is split into three sub-models,
head (H), backbone (B) and tail (T) ( Fig. 1b). Next, we detail
some of the definitions utilized in the framework of SL.
Head (H) resides on the client side and receives input data
D directly from the client’s local dataset. The output of the
head, the smashed data, is sent to the server during forward
propagation. Later, the client receives the gradients from the
server to complete the backpropagation and update the head.
Backbone (B) forms the central part of the model and resides
on the server side. It accepts the smashed data from the head
during forward propagation, further processes it, and sends it
to the T. During backward propagation, it accepts gradients
from T, continues the processes, and sends its output to the
H to complete the training.
Tail (T) is the last part of the model, residing on the client
side. It receives intermediate features from the backbone and
refines them to generate the final output predictions and
backpropagates gradients to the server.
The U-shaped paradigm makes use of the function composition
property that allows obtaining the exact behavior of
F(X) through the implicit concatenation of H, B, and T, as
shown below:
(T ◦ B ◦ H)(X) ≡ F(X)
Given that the composition of functions is always associative
[52], we can show that feeding the head output as
backbone input and giving the backbone output as tail input
still represents the same behavior as the original DNN that the
clients want to train.
T((B ◦ H)(X)) = (T ◦ B ◦ H)(X)
In a standard training scenario, as depicted in Figure 2,
client i passes its data through its head (step 1), then gives the
output of the head to the server (step 2), passes it through its
backbone, and sends the output back to client i (step 3), who
feeds this last output to its tail and calculates the loss using the
ground-truth labels (step 4). During backpropagation (step 5),
the client first calculates the gradients for the tail and passes
the computed gradients to the server, which again computes
the gradients for the backbone using the information provided
by client i. The server then applies backpropagation and passes
the gradients to client i, who applies backpropagation to obtain
the gradients for the head. After these steps, client i shares
the resulting tail and head with client i + 1 (step 6), who
uses them as starting weights for its training step. As the
server does not have direct access to the client’s head and
tail nor to the client’s data, SL enables resource-constrained
devices to train and apply large models in a secure and privacypreserving
[53], [47], [20].
III. PROBLEM SETTING
In this section, we describe the considered system (Sect. III-A)
and characterize the threat model (Sect. III-B), before describing
inherent challenges in SL in mitigating backdoor
attacks (Sect. III-C). In the appendix, we provide a high-level
overview of DNNs (App. A), poisoning attacks (App. B), and
the eligibility of frequency transformations to detect backdoors
(App. C).
A. System Setting
In the rest of this paper, we consider a system consisting of
N clients holding private datasets that are not shared with
other parties. Coordinated by a central server S, they use
the SL framework to train a DNN on their private datasets
collaboratively. An example of such a system is visualized in
Fig. 2. Following existing literature [18], [32], [39], [68], we
focus on a U-shaped SL configuration that splits the DNN into
three parts (Head - H, Backbone - B, Tail - T), where H and
T are located on the client side, whereas B is executed on the
server, as described in Sect. II-A. Since the last part of the
DNN (T) is on the client side, the clients are also responsible
for the loss calculation. Once the training is finished, the client
signs the current model and forwards it with the previous
clients’ models to the next client for the following training
iterations.
For example, the clients could be low-performance devices
such as smartphones that want to train and perform inference
on a large DNN (e.g., a Large Language Model) without
revealing their private training or inference data. The U-shape
enables them to outsource the computation-intensive part to a
high-performance server without violating the data’s privacy.
B. Adversary Model
We consider an adversary A that aims to inject a backdoor
into the collaboratively trained model F. Thus, the backdoored
model F∗ shall predict a specific, adversary-chosen target label
LA when the model receives input x∗ containing the trigger R.
3
𝐻𝐻 𝑇𝑇
𝐶𝐶1
ℒ
𝐶𝐶2 𝐶𝐶3
𝐻𝐻𝑇𝑇
𝐻𝐻𝑇𝑇
1
2
3
4
5
5 B
𝐻𝐻 𝑇𝑇 𝐻𝐻 𝑇𝑇
6
𝐷𝐷1 𝐷𝐷2 𝐷𝐷3
𝐶𝐶N
𝐻𝐻 𝑇𝑇
𝐷𝐷𝑁𝑁
𝐻𝐻𝑇𝑇
𝐻𝐻𝑇𝑇
Fig. 2: Overview of a Split Learning (SL) system that utilizes data from mobile devices but executes the computation-heavy
backbone (B) on a cloud server, while all clients C1, . . .CN provide the data Di, hosts the head T and tail T, as well as
calculates the loss L.
When given a clean input without the trigger, the backdoored
model must generate the correct prediction to prevent the
backdoor from being detected. More formally:
F∗(x) =
(
LA R ∈ x
F(x) R ̸∈ x
(1)
Notably, for a triggered input sample x∗, the clean prediction
differs from the backdoor target label F(x∗) ̸= LA.
A is assumed to have complete control over one or several
malicious clients and can, therefore, arbitrarily manipulate the
input data and labels. Further, due to the considered U-shape
architecture, A can arbitrarily change the local head and tails,
the smashed data and gradients sent by the server, and the loss
function. Aligned with existing work on mitigating backdoor
attacks in distributed learning [44], [8], [25], [27], [41], we
assume A to control at most N/2 − 1 clients. Additionally,
we assume A knows the defense mechanism deployed on the
server side. Thus, A can constrain the training loss utilizing
the metrics used by the defense mechanism.
In the following, we will focus on the attacks that malicious
clients perform. The server has an intrinsic motivation to
produce a well-trained and effective model, as its reputation
is based on the quality of the resulting model. Additionally,
while the client devices are mostly anonymous, as they are
just mobile devices, the server is identifiable and accountable
for its behavior. Since existing literature already investigated
the problem of defending against backdoor attacks that are
conducted by malicious servers [13], we will consider these
attacks to be out of the scope of this paper.
C. Objectives and Challenges
This section details the objectives of a defense that aspires
to prevent backdoor attacks and the challenges encountered in
designing such a defense in the framework of SL.
An effective and practical backdoor defense aims to fulfill
the following security objective:
O1 Prevent Backdoor Attacks: The primary requirement of
an effective defense strategy is to efficiently prevent malicious
clients from injecting a backdoor (as described in App. B)
into the trained model.
However, to ensure that the defense is practical and does
not render the resulting model unusable, the defense must
also fulfill the following functional requirement:
O2 Preserve Model’s Utility: The defense must not
negatively affect the accuracy of the model on the benign
main task (Main Task Accuracy, MA).
This dual focus guarantees the core functionalities, resilience,
and reliability.
Compared to centralized or distributed learning settings such
as Federated Learning, poisoning defenses in split learning
face several unique challenges.
C1: No Data Access: The training data are located on the
client side, preventing the server from inspecting them to
detect manipulated training samples. Especially for scenarios
that involve sensitive data, it is impractical to assume that
clients share their data with the server. Therefore, the server
can detect the backdoor only by analyzing the model updates.
C2: Non-Comparable Client Models: Another challenge
for detecting poisoning attacks in SL is that the models
(updates) of different clients cannot be directly compared to
each other. In SL, clients train in sequential order, and each
client uses the trained model of its predecessor as the base
model for its training. Therefore, the resulting models of two
clients will always differ, even if both clients used the same
data. Additionally, strategies frequently adopted in Federated
Learning [17] to compare the models’ updates cannot be
transferred to SL. Due to the non-convex training process of
DNNs, two clients with the same data might obtain different
model updates if they start training from different base models.
C3: Sequential Training: Sequential training also poses a
4
significant challenge in handling detected poisoned models
because they affect the training results of following clients. In
other distributed learning settings, such as Federated Learning,
the defender typically waits until each client has trained its
local model before analyzing the model updates and excluding
suspicious ones [14], [8]. However, the sequential training of
SL means that if a poisoned model is detected at the end of a
training round when every client has finished its training, the
models of all clients that followed the malicious clients would
need to be discarded. Since these clients used a poisoned
model as their base, their models are likely also to contain
the backdoor, making them unusable for the server.
In addition, even if it would be practical to repeat the
training using a different base model assumed to be benign,
a sophisticated adversary might alternate between benign and
malicious behavior to fool the server and avoid being removed
from the pool of assumed benign clients. Then, when the
training process is repeated, the adversary could try to introduce
the backdoor using a client who had previously behaved
inconspicuously. Therefore, even if the training is repeated,
the defense must also be applied during this repetition. On the
other side, in the case of the absence of any attack, the defense
must be able to accept all models if no poisoning attack is
detected. Otherwise, the defense would cause an endless loop
of repeating the training after seemingly malicious clients are
detected and excluded malicious clients, rendering existing
outlier-detection-based techniques [8], [44], [36] impractical
for Split Learning.
IV. RATIONALE FOR STATIC AND DYNAMIC ANALYSIS
In the following, we elaborate on the intuition behind
the two-fold analysis that is employed to efficiently detect
poisoned models.
Static Analysis: The purpose of the frequency analysis is to
measure fine-granular static changes that are performed in the
backbone during the clients’ training. Previous research [40],
[61] has shown that in the early stages of the training, mostly
the lower components of the models’ frequency representation
change. Only with progressing convergence do the high
frequencies start to change significantly. Therefore, the lowfrequency
components are especially affected when a model is
trained for new behavior. However, when injecting a backdoor
that was trained on benign data in advance by earlier clients,
the backdoor behavior is in contradiction with the benign
model’s behavior as the backdoor target label differs from
the benign prediction (see Sect. III-B). Therefore, significant
adaptions of the model’s behavior are necessary, resulting in
high changes for the low-frequency components.
To perform this analysis, we first transform each model
update into the frequency domain using the Discrete Cosine
Transform (DCT) (App. C). The DCT helps break down the
model’s updates into their frequency components, allowing
unusual changes that indicate backdoor behavior to be spotted.
Specifically, we use the two-dimensional DCT (2-D DCT) because
of its energy compaction property, which means it stores
the most important information in the low-frequency components.
This characteristic helps detect significant changes,
such as those introduced by backdoor attacks. Additionally,
the DCT is computationally advantageous over the Discrete
Fourier Transform (DFT) because its output is always in
real numbers, making it simpler and faster to compute. By
transforming the model updates with the DCT and then
calculating the pairwise Euclidean distances between these
frequency representations, we can effectively detect anomalies
that suggest the presence of backdoor behavior.
To convert a backbone’s distances to all other backbones
into a score, we sum up the distances to the closest n/2+1 other
models. As described in Sect. III-B, we assume a majority of
clients to be benign. Thus, a model that does not belong to the
majority is considered malicious during the current rollback.
Thus, by considering only the n/2 + 1 smallest distance
values, we prevent A from manipulating the score calculation,
e.g., by providing manipulated models (so-called canaries) that
increase the scores of benign models to make the selection
of a regular poisoned more likely. By considering only the
smallest majority of scores is considered, we ensure that A
cannot increase the score for benign models.
Rotational Analysis: During training, backdoor attacks
can cause significant and unusual changes in the orientation
or configuration of the model’s parameters. To detect such
anomalies, SafeSplit employs a dynamic analysis by computing
the rotational distance between pre-trained (historical)
server backbone updates. This rotational metric captures the
extent of changes in the model’s parameter space, providing
a robust measure of how the backbone evolves.
The rotational distance metric is designed to quantify the
orientation shifts in the model’s high-dimensional parameter
space. Unlike static analysis, which focuses on the magnitude
and plain difference for parameter updates, dynamic analysis
focuses on the direction and trajectory of these updates. By
analyzing the rotational distance between different backbones,
SafeSplit can identify deviations from typical training patterns
indicative of backdoor behavior. This can be seen as an
interpolation of the path that the optimizer took during training
from the base model Bt−1 to the trained model Bt, thus
revealing information about the training objective.
SafeSplit computes the rotational distance metric RD (more
details in Section V-D) for each server’s backbone. The
reasons for utilizing it in the backdoor detection are three-fold.
First, RD analyzes directional changes. The rotational distance
metric is sensitive to the direction of parameter updates.
Backdoor attacks typically introduce abrupt and significant
directional changes to implant malicious behavior, which can
be detected by observing large rotational distances. Second,
RD is robust to scaling. By focusing on the orientation rather
than the magnitude, the rotational distance metric is robust to
variations in the scale of parameter updates, which might occur
due to changing client-controlled hyperparameters such as the
learning rate, optimization algorithm or loss function. Third,
RD complements the static analysis. While static analysis
detects anomalies based on the magnitude and difference
5
min
2
+1
2 1 3
3
1
2
3
2 1 3
1 3 2
2 1
3 2 1
3
min
2
+1
( 2, 1 , , 3 )
3
3
Fig. 3: Workflow of SafeSplit to skip or poisoned models based on an analysis of the models in the frequency domain and
their rotational displacement. The workflow is shown for an example scenario consisting of 4 clients.
of updates in the frequency domain, dynamic analysis adds
a layer of scrutiny by examining the trajectory of these
updates. Together, they cover both perspectives and provide
a comprehensive defense mechanism.
Another important aspect of SafeSplit is the circle-wise
analysis that ensures using the latest benign-detected model
and, therefore, skipping the malicious clients. This prevents
benign clients from training on poisoned models. This approach
strengthens the security of the training process, ensuring
model reliability as we do not remove the benign clients
that have been misclassified as malicious.
V. SAFESPLIT
In the following, we describe the high-level design of SafeSplit
(Sect. V-A) and the underlying intuition (Sect. IV) before
elaborating on its model-skipping mechanism (Sect. V-B): frequency
analysis to detect static changes (Sect. V-C), and rotational
distance analysis to detect dynamic changes (Sect. V-D).
The overall workflow of SafeSplit is visualized in Fig. 3 and
formalized in Alg. 1. Fig. 4 shows the analysis process in more
detail.
A. High-level Design
SafeSplit is deployed on the server and analyzes the model’s
backbone parameters updates. Thus, SafeSplit is executed
before each client’s training starts to select a benign starting
point. The high-level process is shown in Fig. 3. Each time
the backbone parameters are updated after a client finishes its
Determine Neighborhood Euclidean Distance
Consider lowest 𝑁𝑁⁄
Consider lowest ⁄ 2 + 1 values for majority 𝑁𝑁
2 + 1 values for majority
Calculate 𝜃𝜃𝑖𝑖 𝑖𝑖∈[𝑡𝑡−𝑁𝑁+1,𝑡𝑡]
Angular
Displacement
Bt
Bt-N+1, …, Bt
Determine Rotation Displacement Metric
𝑅𝑅𝐿𝐿1 ,…𝑅𝑅𝐿𝐿𝑁𝑁􀵗
2+1
𝑅𝑅𝑡𝑡−𝑁𝑁+1,…𝑅𝑅𝑡𝑡
Backbone
History
Check if current client i is in both majority groups
Roll Back to
previous Model
Continue using
current model
𝑆𝑆 𝑖𝑖∈[𝑡𝑡−𝑁𝑁+1,𝑡𝑡]
Calculate Discrete Cosine Transformation
𝐸𝐸𝐿𝐿1 ,…𝐸𝐸𝐿𝐿𝑁𝑁􀵗
2+1
𝐸𝐸𝑡𝑡−𝑁𝑁+1,…𝐸𝐸𝑡𝑡
Rotational Distance
(Dynamic Changes)
Frequency Analysis
(Static Changes)
Calculate
Angular Velocity
𝜔𝜔 𝑖𝑖∈[𝑡𝑡−𝑁𝑁+1,𝑡𝑡]
Fig. 4: Overview of SafeSplit, using the latest backbone model
Bt and previous backbones Bt−N+1 . . .Bt−1 to determine
Rotation Displacement Metric values Rt−N+1, . . . ,Rt and the
Euclidean Distance Neighborhood Scores, before returning the
index of most recent backbone Li being among N/2+1 lowest
values RL1 , . . . ,RLN/2+1 and EL1 , . . . ,ELN/2+1 .
training and before the next client starts its training (step 1 in
Fig. 3), SafeSplit analyzes the updated backbone for backdoor
behavior (steps 2 - 9 in Fig. 3). If no malicious behavior is
detected, the training process continues by training the next
client using the latest model of the previous client as the base
model (steps 9 and 10 in Fig. 4).
6
However, if backdoor behavior is detected, the rollback
mechanism iterates through the previous clients’ training outputs
to identify the backbone’s latest benign state. Thus, this
rollback mechanism reverts to the previous server’s backbone
state and continues the examination to ensure a benign trained
state is indeed selected, bypassing all the malicious client’s
trained updates (step 8 in Fig. 4). In this case, the current
backbone is replaced by the identified latest benign checkpoint,
and the next client is instructed to train using the respective
head and tail associated with the benign backbone found (step
10 in Fig. 4).
We employ a two-fold analysis strategy to detect the benign
state of the server’s backbone and detect poisoned updates.
First, we perform a static analysis to analyze the Euclidean
distances of the backbones’ frequency representation (steps
5 and 6 in Fig. 4). Second, we perform a dynamic analysis
to quantify the evolution of the backbone’s values using our
novel rotational distance metric (step 4 in Fig. 4). Both metrics
are based on the observation that the backdoor behavior
contradicts the benign behavior. While different benign clients
either have similar training behavior in the case of IID data
or orthogonal behavior in the case of non-IID data, the
backdoor aims to make the model mispredict a wrong label
(see Sect. III-B), to the backdoor target. Thus, the malicious
client needs to change parts of the benign training and train a
new backdoor behavior embedded in the model, resulting in
large distances for the leveraged metrics.
We use both metrics to determine the benign state of the
server’s backbone, as each investigates a different perspective.
The distances of each metric are used to determine a score for
each model that indicates the alignment of the model’s training
objective. A high score indicates that the current model’s
training objective was in contradiction with the behavior of
a majority of other clients. SafeSplit considers a model to be
benign if, for both metrics, the model’s scores are small, thus
within the respective sets of smallest set N/2 + 1 of existing
score values (see lines 17-23 in Alg. 1 and step 7 in Fig. 4).
Since the majority of models are assumed to be benign (see
Sect. III-B), a model that does not belong to the majority is
considered malicious during the current rollback.
B. Circular Benign Model Identification
After each client i completes training in their respective
communication round t with the server (lines 8 - 9 in Alg. 1,
step 1 in Fig. 4), we store the server’s backbone Bt in a FIFO
array (step 2 in Fig. 4. This process is repeated for each training
step t, ensuring we retain the last N models. Once client i
finishes training, we compute the deployed metrics (Rotational
Distance and Euclidean distance of frequency representation)
for all backbone models Bt−N+1, . . . ,Bt stored in the FIFO
array (lines 10-16 in Alg. 1 and steps 3-6 in Fig. 4).
Next, we determine the benign checkpoint B (lines 17-25 in
Alg. 1). This process determines whether the backbone being
examined shows backdoor behavior. For this, the server checks
for each metric if the model’s score is within the smallest
N/2 + 1 of existing score values (lines 17-19 in Alg. 1 and
step 7 in Fig. 4, see also Sect. IV). Alternatively, this process
determines how many clients to skip in backward direction
to reach a benign backbone checkpoint, i.e., obtain B and
the index of the corresponding optimal head (H) and tail (T)
of the client determined showing benign behavior (step 8 in
Fig. 4). Afterward, the next client is informed which head and
tail to use as base model for it’s training (step 10 in Fig. 4).
It should be noted that we describe the circular benign
model identification for a system where only the backbone
is held on the server, while the head and tail always reside
on the client side and are forwarded from each client to its
successor. As described in the system setting (Sect. III-A),
each client signs its training result to allow the succeeding
clients to verify the origin of the respective used head and
tail. This ensures that the server never gets access to the head
and tail, leading to improved privacy. However, SafeSplit can
be straightforwardly adapted to scenarios where each client
sends the head and tail after the training to the server, and the
server forwards them to the next clients.
C. Frequency Variation Computation
After each client i finishes training, we use the Discrete
Cosine Transformation (DCT) to determine the frequency
representation of each model and select the low frequencies
(line 10 in Alg. 1 and step 5 in Fig. 4). Then, we compute the
Euclidean distance in the frequency domain for the server’s
backbone parameters (line 11 in Alg. 1). Let Bt represent
the backbone parameters of the server at the training step t.
The low-frequency component of the backbone’s update St is
computed as:
St = DCTlow(Bt − Bt−1) (2)
where DCTlow(Bt) denotes only the low-frequency components
of the Discrete Cosine Transform (DCT) applied to
the backbone’s update Bt −Bt−1. We then calculate for each
frequency representation St the Euclidean distances to each
other frequency representation (step 6 in Fig. 4):
edt,i = ∥St − Si∥2 (3)
The frequency score Et of the model Bt is then calculated
as the sum of the distances to the frequency representation of
the N/2+1 closest other models (line 12 in Alg. 1 and step 6
in Fig. 4).
The DCT converts the backbone parameters Bt into their
frequency domain representation. This transformation allows
us to analyze how the frequency components of the backbone
parameters change over time. The Euclidean distance Dt
quantifies the pairwise differences between the frequency representations
of the backbone parameters for each client i and
j in (Bt−N+1,Bt). Larger distances may indicate significant
changes in the backbone’s frequency characteristics, which
could be indicative of injecting contradicting behavior, being
typical for backdoor attacks. This static analysis provides
insights into the stability and consistency of the server’s backbone
parameters over multiple training steps, helping to detect
and mitigate potential security risks in the SL framework.
7
D. Measuring Rotation Distances
In this section, we describe the computation of the rotational
distance metric (step in Fig. 4) that captures the
dynamic changes in the server’s backbone’s update for each
client’s training. As mentioned in Sect. V-B, to determine
the checkpoint of the benign behavior, we keep track of
the N historical backbone updates and compute the pairwise
differences between RD of the backbone parameters for each
client i, j ∈ {Bt−N+1,Bt} after training the current client i.
RD computation consists of three steps: First, we compute the
angular displacement of the parameter values in the backbone
(line 13 in Alg. 1). Second, we compute the angular velocity
to quantify the rate of change in the angular displacement.
Lastly, we compute the rotational frequency to identify the
frequency of these directional changes.
Angular Displacement Computation. The angular displacement
θ(t) measures the orientation change between successive
backbone updates. To inject contradicting backdoor behavior,
the attack introduces significant directional changes, causing
θ(t) to deviate from normal training patterns. Thus, given the
current client i and its backbone model Bi, we compute the
angular displacement that measures a rotating object’s change
in orientation (line 13 in Alg. 1).
Let Bt be a backbone at step t. The angular displacement
θ(t) is given by2:
θ(t) = arctan (Bt) (4)
We compute θ(t) for each client i in (Bt−N+1,Bt) after
training the current client (see App. H for details). This allows
to obtain the angular position at each time point. The angular
displacement θ(t) measures the change in orientation between
successive backbone updates. Backdoor attacks often introduce
significant directional changes, causing θt to deviate from
normal training patterns.
Angular Velocity Computation. The rate of change of the
angular displacement over time, or angular velocity ω(t), is
given by:
ω(t) =
θ(t) − θ(t − 1)
Δt
(5)
where Δt is the time interval between successive updates.
The angular velocity ω(t) quantifies the rate of change in θ(t)
(see line 14 in Alg. 1). High angular velocities ω(t) indicate
rapid shifts in the parameter space, which are uncommon
during regular training but typical of backdoor insertion
attempts.
Rotational Distance Metric (RD) Computation. At the
end, we compute the rotational displacement metric RD,
which is also the rotational frequency that captures the frequency
of these directional changes (see line 15 in Alg. 1).
RD is defined as:
2In an earlier version of this manuscript, by accident, ”arccos” instead of
”arctan” was written.
RD =
ω(t)
2π
(6)
We divide ω(t) by 2 · π because one complete revolution
(or cycle) around a circle corresponds to an angle of 2 · π
radians. So, dividing by 2·π allows us to convert ω, measured
in radians per second, into the rotational frequency, typically
measured in cycles per second or Hertz (Hz).
The rotational frequency RD, derived from ω(t), identifies
the frequency of these directional changes. A high RD
suggests frequent and abrupt alterations in parameter
orientation, which is characteristic of backdoor attacks that
aim to alter model behavior.
The rotational distance metric allows capturing dynamic
changes in the orientation or configuration of the backbone
during the local training of clients. It primarily measures
angular changes between gradient vectors (geometrically). As
we show in the following sections, this makes the metric
highly effective for identifying deviations of poisoned models.
A major strength of the metric is its ability to analyze the
rotational changes over time, making it resilient to small,
random perturbations or noise introduced in the gradients.
Hence, it is able to discern the changes between the malicious
updates even if the noise is systematic and mimics the patterns
introduced by a poisoning attack. This makes the metric robust
against manipulations to hide the poisoned gradients.
After the DCT and RD computation, we use them to compute
the benign checkpoint to superimpose the checkpoint’s
backbone model B∗ to the current backbone state Bi at the
server and transfer the corresponding optimal head (H∗) and
tail (T∗) of the client at the benign checkpoint to the next
client for its training (to use as the base model).
VI. EVALUATION
This section presents a comprehensive evaluation of
SafeSplit across various types of backdoor attacks, showcasing
our defense mechanism’s ability to maintain backdoor accuracy
below 5%. In App. D, we analyze the effectiveness of
the combination of static and dynamic perspectives. In App. F
we evaluate further non-backdoor attacks.
A. Experimental Setup
Metrics: Our evaluation of SafeSplit leverages the following
key metrics.
Backdoor Accuracy (BA) indicates the model’s accuracy
concerning malicious tasks. It quantifies the fraction of the
trigger set for which the model generates predictions aligned
with the attacker’s objectives. The attacker aims to maximize
BA, while SafeSplit strives to minimize it.
Main Task Accuracy (MA) measures the model’s accuracy
for benign tasks. It reflects the percentage of clean inputs for
which the model delivers accurate predictions. The adversary
aims to reduce the impact on MA to diminish the likelihood
of detection. An essential requirement of SafeSplit is not to
8
Algorithm 1 SafeSplit
1: Input: N, R, B0, H0, T0, clients ▷ N is the number of clients, R the number
of training rounds, H0,B0, T0 build the initial model, a FIFO list of N clients
2: Output: B∗, H∗, T∗ ▷ updated backbone, heads and tails
3: function SMALLESTMAJORITY(v1, . . . , vN)
4: return sorted(v1, . . . , vN)[1,. . . ,N/2 + 1]
5: end function
6: H,B, T ← H0,B0, T0
7: for each training step t ∈ [1,R · N] do
▷ Perform Training
8: current client ← clients[t mod N]
9: H∗
t ,B∗
t , T∗
t ← TRAIN(current client,H∗,B∗, T∗)
▷ Measure Distance in Frequency Domain
10: ∀i ∈ {t − N + 1, . . . , t}Si ← DCT(Bi − Bi−1)
11: ∀i, j ∈ {t − N + 1, . . . , t}edi,j ← EuclideanDistance(Si, Sj)
12: ∀i ∈ {t − N + 1, . . . , t}Ei ←
P
SmallestMajority(edi,t−N+1, . . . , edi,t)
▷ Calculate Rotational Distance
13: ∀i ∈ {t − N + 1, . . . , t}θi ← AngularDisplacement(Bi)
14: ∀i ∈ {t − N + 1, . . . , t}ωi ← AngularVelocity(θi)
15: ∀i ∈ {t − N + 1, . . . , t}RDi ← RotationalDisplacement(ωi)
16: ∀i ∈ {t − N + 1, . . . , t}Ri ←
P
SmallestMajority(i,RDt−N+1, . . . ,RDt)
▷ Determine Benign Majority
17: frequency majority = arg sort(Et−N+1, . . . ,Et)[1, . . . , N/2 + 1]
18: rotation majority = arg sort(Rt−N+1, . . . ,Rt)[1, . . . , N/2 + 1]
19: benign majority ← rotation majority ∩ frequency majority
▷ Determine Latest Benign Model
20: for each previous client c ∈ [t, max(t − N + 1, 1)] do
21: if c ∈ benign majority then
22: H∗,B∗, T∗ ← Hc,Bc, Tc
23: break
24: end if
25: end for
26: end for
27: return H∗,B∗, T∗
significantly affect the MA.
Datasets. Aligned with existing work on SL [65], [22],
[38], [64], [6], [50], [19], [13], [51], [16], we leveraged
five datasets (CIFAR-10, FMNIST and MNIST, GTSRB,
CIFAR-100) to perform our experiments:
CIFAR-10 consists of 50 000 training and 10 000 test images of
size 32×32 pixels, showing objects and animals belonging to
10 different classes [26]. As DNN, we use the widely adopted
ResNet-18 architecture.
MNIST consists of 60 000 training and 10 000 test grayscale
images showing handwritten digits. Aligned with recent work
on distributed learning, we implemented a Convolutional Neural
Network (CNN), as described by Cao et al. [10].
FMNIST is composed of 60 000 training and 10 000 test
images, each sized 28x28 pixels, depicting various types of
clothing across 10 classes [59]. As DNN we use also the CNN
described by Cao et al. [10].
CIFAR-100 consists similar to CIFAR-10 of 50 000 training
and 10 000 test images but is categorized into 100 classes.
Due to the high number of labels, which are significantly
larger than the considered number of clients. As DNN, we
used Wide-ResNet50, being pretrained for ImageNet dataset
and replaced the final layer, while the training included all
layers.
GTSRB consists of 39 000 training and 12 600 test images
of varying sizes, from 32 × 32 to 64 × 64 pixels, showing
traffic signs belonging to 43 different classes [48]. We used
the MicronNet [57] architecture as DNN.
Computational Setup. We conducted the experiments using
TABLE I: Overview of used Deep Neural Network (DNN)
Architectures.
Number of Parameters
Model Head Backbone Tail Total Evaluated
Datasets
ResNet18 [21] 9 536 11 166 976 5 130 11 181 642 CIFAR-10
Simple CNN [10] 1 520 665 562 5 130 672 212 CIFAR-10
Simple CNN [10] 520 435 162 5 130 440 812 MNIST, FMNIST
GoogLeNet [49] 124 736 5 475 168 10 250 5 610 154 CIFAR-10
VGG11 [46] 1 920 9 224 064 4 199 946 13 425 930 CIFAR-10
Wide-ResNet50 [66] 9 536 66 824 704 204 900 67 039 140 CIFAR-100
MicronNet [57] 824 411 192 3 010 415 026 GTSRB
the deep learning library PyTorch [3]. The experiments were
conducted on a server with 4x NVIDIA A6000, an AMD
EPYC 7773X CPU using 64 physical cores, and 768 GB of
main memory.
Model Architectures. Due to the absence of existing work
on mitigating backdoor attacks on SL, we aligned the used
DNN architectures on existing work about the security of
Federated Learning [10], [14]. To ensure a comprehensive
evaluation, we included 4 different model architectures in
our evaluation with parameter sizes ranging from 440 812 to
13 425 930 trainable parameters. The details of the used DNN
architectures are shown in Tab. I. We focused our evaluation on
the CIFAR-10 dataset and evaluated all DNN architectures for
this dataset (see Sect. VI-H). In addition, we also evaluated
the MNIST and FMNIST datasets. Due to the simplicity of
these datasets, we used a simple CNN for these datasets, as
defined by Cao et al. [10] for their evaluation of backdoor
attacks against Federated Learning. Notably, the structure of
this simple CNN slightly varies for different datasets, as the
images in CIFAR-10 have a dimension of 32×32 pixels while
MNIST and FMNIST consist of images with the dimensions
28 × 28.
Training Parameters. Unless stated otherwise, we considered
a system consisting of 10 clients, from which 2 are malicious
and aim to inject a backdoor. We simulated non-IID data on
the client side using the main-label strategy that is frequently
used in other work on distributed learning [9], [14]. Here,
each client gets randomly assigned a main label. While a
certain fraction, indicated by the IID rate, is samples from
all available samples, the remaining samples are chosen only
from the assigned main label class. An IID rate of 1.0 indicates
a complete IID data distribution among clients. As the default
value, we use an IID rate of 0.8. Only for GTSRB and CIFAR-
100 we used an IID-rate of 1.0 as default value to achieve a
decent MA, as the number of clients here is significantly lower
than the number of labels.
Considered Backdoor Behavior. In the following, we evaluate
SafeSplit’s effectiveness for different datasets and backdoors.
Unless stated otherwise, we use for CIFAR-10 a semantic
backdoor that misclassified cars in front of a striped
background as birds and for all other data sets a backdoor
being activated by a red rectangle or, in the case of grayscale
data sets such as MNIST, a white rectangle.
9
B. Effectiveness of SafeSplit
SafeSplit undergoes extensive evaluation against different
backdoor attacks for pixel and semantic triggers, as depicted in
Table II. As the table shows, SafeSplit reduced the BA in cases
to a negligible value. Notably, for pixel trigger backdoors,
the BA is often even for a benign model not exactly 0%.
This phenomenon occurs because the model misclassified
some test samples, and the metric counts them in favor of
the backdoor if the predicted label is equal to the backdoor
target label (see App. E for details). Furthermore, we conduct
thorough evaluations on the CIFAR-10 dataset in various
attack scenarios, which we describe in the following.
C. Different Data Scenarios
We conducted different experiments on the CIFAR-10
dataset, varying the degree of non-IID data distribution to
assess its effect on SafeSplit performance. The results are
shown in Table III. Across IID rates of 0.6, 0.8, and 1.0,
SafeSplit demonstrates significant reductions in BA while
maintaining MA. Notably, for decreasing IID rates, the MA
goes down regardless of the presence of an attack or defense,
as training in such non-IID settings becomes very challenging.
Further, we assessed different numbers of clients being involved
in the training process, ranging from 5 to 20, with 20%
of them being malicious. Fig. 5 illustrates the BA and MA for
CIFAR-10. As the figure shows, SafeSplit effectively mitigates
the attack, therefore keeping the BA 0% and maintaining the
MA with only a negligible drop compared to the scenario
without a defense applied.
D. Impact of Poisoned Model Rate (PMR)
Fig. 6 shows SafeSplit’s effectiveness for varying ratios
of malicious clients (PMR). As the figure shows, SafeSplit
effectively mitigates the backdoor, keeping the BA at 0%,
while maintaining the MA close to the scenario without
defense. Notably, the figure also shows that the attack is less
effective for low PMRs.
TABLE II: Effectiveness of SafeSplit against different attacks
for the respective dataset, in terms of Backdoor Accuracy (BA)
and Main Task Accuracy (MA).
Attacks No Defense SafeSplit
Dataset BA MA BA MA
CIFAR-10 Semantic Trigger 59.3 66.6 0.0 62.7
Pixel Trigger 100.0 66.6 0.3 66.4
MNIST Pixel Trigger 86.2 98.7 0.0 98.8
FMNIST Pixel Trigger 79.8 83.0 3.4 84.6
CIFAR-100 Pixel Trigger 93.3 76.8 0.1 76.5
GTSRB Pixel Trigger 30.0 58.0 0.6 63.7
TABLE III: Effectiveness of SafeSplit for different degrees of
IID data.
No Defense SafeSplit
IID-Rate BA MA BA MA
0.6 44.7 59.5 0.0 57.5
0.8 59.3 66.6 0.0 62.7
1.0 60.7 69.4 0.0 68.1
E. Adaptive Attacks
A sophisticated adversary that is aware of the defense might
adapt the attack to make it more effective against the deployed
defense approach. In the following section, we describe and
evaluate various sophisticated attack strategies. Notably, some
of the adaptive attacks assume a stronger adversary having
knowledge of the current parameters of the backbone model,
exceeding the previously defined threat model (see Sect. III-B).
In practice, such an adversary could, for instance, approximate
the backbone’s parameters using a shadow model trained
separately with the malicious clients’ benign data. However,
for the sake of a comprehensive evaluation, the following
section assumes that the adversary has knowledge of the actual
backbone parameters.
Varying the Poisoned Data Rate (PDR). To increase the
similarity of the poisoned models to benign models, A might
vary the ratio of data samples for the backdoor behavior in
its local dataset (Poisoned Data Rate, PDR). The choice of
this parameter realizes a tradeoff since low PDRs ensure that
the resulting model remains inconspicuous but also reduce the
efficiency of the attack, while high values result in a high
attack impact but also make the models easier to detect. Tab. V
shows the effectiveness of SafeSplit for varying PDRs. As the
table shows, SafeSplit effectively mitigates the attack for all
PDR values. Notably, the table also shows that the attack is,
in the absence of a defense, most effective for a PDR of 50%,
while for higher PDR values, the BA is reduced. Although
this might be counterintuitive at first glance, a reason for this
might be that for high PDRs, the dataset of the malicious
clients becomes highly imbalanced and focuses on the single
backdoor target label. This imbalanced data causes the model
to overfit significantly and results in changes that can be easily
reverted by the benign clients through their training.
Loss Constrain for Rotation Distance Metric. A key aspect
of SafeSplit’s backdoor detection is the rotational distance
metric. Following existing attacks on Federated Learning [5],
a sophisticated adversary might adapt the loss function and
integrate a regularization term into the loss function that
minimizes the rotational distance of the current poisoned
model to a reference model. Given the loss function Lclass that
measures the model’s performance on its training data, the
anomaly evasion loss Lano that measures the suspiciousness
of the model, here the rotational distance to the reference
5
10
15
20
25
30
Number of clients
0%
20%
40%
60%
80%
100%
Accuracy
BA SafeSplit
MA SafeSplit
BA No Defense
MA No Defense
Fig. 5: BA and MA for different participant numbers.
10
model, then according to Bagdasaryan et al. the combined
loss function is defined as
L = α · Lclass + (1 − α) · Lano (7)
where α is a hyperparameter that weights both terms.
We evaluated this attack using the base model (A has
knowledge of the backbone) as reference model. Notably,
knowing the backbone is not a realistic threat scenario, as
the adversary would need to have access to the server, going
beyond our threat model (see Sect. III-B). Tab. IV shows
the results for different α-values. Although the adversary
reduces the suspiciousness for the rotational distance, due to
the combination of a static and dynamic analysis SafeSplit still
effectively detected the poisoned models and reduced the BA
to 0%. Notably, setting α = 0.5 achieves in the absence of a
defense the best BA without significantly degrading the MA.
Thus, in the following experiments, we use α = 0.5.
TABLE IV: Effectiveness of SafeSplit against loss-constrain
using different α-values.
No Defense SafeSplit
BA MA BA MA
α = 0.25 62.0% 66.3% 0.0% 63.5%
α = 0.50 60.0% 66.4% 0.0% 64.1%
α = 0.75 51.3% 65.9% 0.0% 63.9%
To evaluate also a more realistic setting, we repeated
the experiment using a separate backbone as reference that
was trained from the same base model. However, also here
SafeSplit was able to detect the poisoned model updates,
resulting in a BA of 0%.
Loss Constrain for Static Distance Metric. Analogous to the
previous paragraph we added a regularization term that focuses
on the frequency analysis of SafeSplit as regularization term.
Again, we evaluated the attack using the base model and also a
benign model as reference models. In both cases, we extracted
the low frequencies of the Discrete Cosine Transform of the
reference model and the poisoned model under training; we
then constrained the loss function to minimize the Euclidean
Distance between the two frequency representations. However,
due to the characteristics of the Discrete Cosine Transform
domain, even small changes in the low-frequency components
of the model can result in significant changes in the model’s
parameters. The regularization terms of the loss function, can
10%
15%
20%
25%
30%
35%
40%
Poisoned Model Rate (PMR)
0%
20%
40%
60%
80%
100%
Accuracy
BA SafeSplit
MA SafeSplit
BA No Defense
MA No Defense
Fig. 6: BA and MA for various Poisoned Model Rates (PMRs).
either try to only inject the backdoor behavior (with α values
close to 1) or try to bring the malicious model DCT as
similar as possible to a benign one (with α values close to
0). Therefore, in the first case, the model behaves again as
a malicious model, and it is detected by SafeSplit, and in
the second case the adversary fails to implant any meaningful
backdoor, with a BA of 0%. A third case is a balance of
the two regularization terms (with α values close to 0.5), but
in this scenario, the changes to the low frequencies produce
a scrambled model with very unnatural behavior. Therefore,
as SafeSplit is highly sensitive to dynamic changes in model
behavior, it was still able to detect the backdoored models and
reduced the BA to 0% in both cases.
Loss Constrain for Static and Dynamic Distance Metrics.
Building on the two previous attacks, we integrated both
loss functions as regularization terms. However, we observed
SafeSplit to remain effective and reduce the BA to 0% for both
reference models (base model and trained benign model). This
might be caused by the trade-off that the DNN optimizer needs
to perform during the training. Because of the regularization
term, the model must not show any indications of contradicting
(backdoor) behavior. However, at the same time, due to the
original loss that focuses on optimizing the predictions, the
model is trained to show backdoor behavior. The optimizer
then aims to minimize both aspects and perform a trade-off
between both aspects. However, the resulting model will then
still show poisoned behavior, allowing SafeSplit to detect it.
Loss Constrain for Euclidean Distance. An alternative
option is to constrain the model not with respect to the specific
metric that the defense uses but to integrate the Euclidean
distance into the loss function to keep all parameters of the
model close to the respective reference model. However, as
this technique needs to perform a trade-off again, it still needs
to train the model on the backdoor behavior, SafeSplit detected
the nuances that indicate the poisoning and reduced the BA
to 0%.
Focus Training on Tail. SafeSplit analyses only the backbone
model, as the head and tail are held by the clients, and
the server has no access to them in order to preserve the
client’s privacy. However, a sophisticated adversary might try
to exploit this and train the backdoor only into the tail. To
achieve this, the malicious clients train the model alternating
with batches containing only benign samples and samples
also containing samples for the backdoor behavior. For the
batches containing backdoor behavior, the client uses the
server for forward propagation, thus making the predictions.
However, after calculating the loss for the poisoned batches,
TABLE V: Effectiveness of SafeSplit for different Poisoned
Data Rates (PDRs).
No Defense SafeSplit
PDR-Rate BA MA BA MA
25% 85.3 68.4 0.0 64.7
50% 88.7 67.9 0.0 65.1
75% 59.3 66.6 0.0 62.7
100% 24.0 51.0 0.0 64.5
11
No Defense
KRUM
Differential Privacy
FreqFed
SafeSplit
0%
50%
100%
Backdoor Accuracy
(a) Backdoor Accuracy
No Defense
KRUM
Differential Privacy
FreqFed
SafeSplit
0%
50%
100%
Main Task Accuracy
(b) Main Task Accuracy
Fig. 7: Comparison of different state-of-the-art defense techniques
against SafeSplit using Main Task Accuracy (MA) and
Backdoor Accuracy (BA).
the backpropagation and gradient descent are only applied to
the tail, and the gradients are not shared with the server. As a
result, the backbone and head will be trained only with clean
data, while the tail is trained for benign and poisoned data. In
this case, the backbone would be inconspicuous. However, as
our experiments showed, for this attack strategy, the backdoor
is not injected into the model, even without defense, and the
BA remains 0%.
F. Comparison with Defenses for Federated Learning
While the existing literature does not provide defenses
against client-side backdoor attacks in SL, many defenses
were proposed for the distributed learning scheme Federated
Learning. Based on the categorization of Fereidooni et al. [14],
we selected 3 representative backdoor defenses for FL and
adapted them for SL. Particularly, we evaluated FreqFed [14]
as an approach that aims to classify all benign and poisoned
models, KRUM [8] that focuses on correctly identifying a
subset of benign models, and Differential Privacy [5], [35] that
aims to mitigate the backdoor without identifying the attackers.
Notably, defending backdoor attacks in SL raises significant
challenges (see Sect. III-C), which SafeSplit addresses. To
avoid any unjustified disadvantages for the existing techniques,
we adapted them, using Euclidean distance of the updates
for Krum and integrated SafeSplit’s circlewise structure into
FreqFed. For Differential Privacy no adaptions were necessary,
as noising and clipping are independent of the SL’s structure.
Fig. 7 shows the BA and MA values for a non-IID scenario
using an IID rate of 0.6. All defenses except Differential
Privacy maintain a comparable MA. However, despite the
adaptions, only SafeSplit is able to mitigate the attack and
reduce the BA to 0%.
G. Performance in the Absence of Attacks.
An important objective for a practical defense is to not
negatively affect the training process (see O2 in Sect. III-C).
To measure the impact of SafeSplit on the training of a model,
we conducted several experiments starting from a random base
when no attack is performed and no defense is applied or
SafeSplit is deployed. After 50 rounds of training without
defense, the MA reached 69.30%, while when SafeSplit was
used, the MA achieved 66.6%; thus, also, when deploying
SafeSplit, no significant drop in MA was observed.
H. Performance for Different Model Architectures.
We evaluated the effectiveness of SafeSplit using different
model architectures. Tab. VI shows the effectiveness of
SafeSplit for 4 different model architectures that are frequently
used in distributed learning [10], [14], [20]. As the table shows,
SafeSplit successfully mitigates the backdoor for all models
while maintaining the MA on a similar level as without the
attack.
In addition, we also evaluated different positions for the
cutting layer for the ResNet-18 model to simulate different
backbone sizes. Resnet-18 consists of a convolutional layer
a batch normalization layer, a ReLu layer, a max-pooling
layer, 4 blocks and a linear layer. Each block consists of
4 convolutional layers, several batch normalization layers,
and a down-sampling layer. We simulated different backbone
sizes by assigning different numbers of blocks (2,3,4) to the
backbone and clients. We observed SafeSplit to effectively
mitigate the attack and reduce the BA to 0%.
In summary, we evaluated SafeSplit for different datasets
and data scenarios, attack types and settings, and client
numbers and compared it against different baseline defenses.
SafeSplit was always able to mitigate the backdoor attack and
reduced in all experiments the BA to less than 5%.
VII. SECURITY CONSIDERATIONS
In the following, we discuss the security aspects of our scheme
to fulfill our security and functional objectives and challenges
(cf. Sect. III-B).
To address objective O1 (Prevent Backdoor Attacks), a
backdoor defense must fulfill the security requirement of
significantly reducing the attack impact. In the following,
we will first discuss the risk of server-side attacks, before
summarizing SafeSplit’s resilience against backdoor attacks
(Backdoor Resilience), including its robustness against
defense-adapted attack strategies that aim to make the DNNs’
parameters inconspicuous (Adaptive Attacks) as well as
attack strategies that aim to prevent that the actually poisoned
parameters are analyzed by the defense (Analysis Evasion).
Effectiveness against Different Adversary Models. We introduced
the first defense against backdoor attacks in Split
Learning, focusing on client-side attacks. As mentioned in
Section III-B, server-side backdoor attacks have already been
the subject of investigation in the literature [13]. In contrast,
12
TABLE VI: Effectiveness of SafeSplit for different Deep
Neural Network Architectures using the CIFAR-10 dataset.
No Defense SafeSplit
No Attack Attack Attack
Model BA MA BA MA BA MA
ResNet-18 0.0 69.3 59.3 66.6 0.0 62.7
Simple CNN 0.0 64.1 78.0 62.7 0.0 60.4
GoogLeNet 0.0 63.5 16.7 57.9 0.0 60.2
VGG11 0.0 47.6 76.7 49.7 0.0 43.0
client-side backdoor attacks remained an open challenge, as
we discussed in Section III-B. Moreover, in typical practical
settings, we expect the server side to be a well-protected cloud
instance, unlike the client side, which can be any mobile device
with less protection. In Sect. VI, we extensively evaluated
SafeSplit in various scenarios using different attack settings.
Backdoor Resilience. In the Sect. VI, we evaluated the effectiveness
of SafeSplit in various scenarios and attack settings.
Particularly, we used five state-of-the-art image recognition
benchmark datasets (CIFAR-10, MNIST, FMNIST, GTSRB,
CIFAR-100; see Tab. II), different numbers of clients (see
Fig. 5), IID settings (see Tab. III), 4 different model architectures
(see Tab. VI) and various scenarios. In addition,
we evaluated different attack settings, particularly different
ratios of poisoned data (see Tab. V) and attack strategies (see
Sect. VI-E). We observed that in all cases SafeSplit mitages
the attack successfully, thus achieving O1 (see Sect. III-B).
Adaptive Attacks. A powerful adversarial evasion strategy
concerns manipulating the training process to ensure the
analyzed backbone model does not show high angular distance
values. In this paper, we consider a sophisticated adversary,
being aware of the defense and having full control over
the individual clients (see Sect. III-B). In Sect. VI-E, we
evaluated several defense agnostic attacks that aim to circumvent
SafeSplit by leveraging loss-constrain and integrating the
angular distance and frequency distance into the loss function,
restricting the distance to benign models, and training the
backbone only with benign data. However, SafeSplit was
able to successfully mitigate all of these attacks, showing
SafeSplit’s robustness even against sophisticated adaptive attack
strategies.
Analysis Evasion. In our adversary model, the adversary has
full control over the head and tail (see Sect. III-B) and can,
before forwarding them to the next client, even entirely replace
the values. Thus, the forwarded values are not necessarily
those obtained by the training. SafeSplit counters this attack by
analyzing the backbone stored on the server, where they can
only be changed in a well-controlled manner (backpropagation).
This ensures that the model used for the following clients
is the same for backdoor detection, effectively preventing
time-to-check-time-to-use attacks. Another strategy to distract
angular metrics in other distributed settings is upscaling the
model’s parameters without changing them [5], [7]. However,
as only the server has access to the backbone, this strategy
is restricted to the head and tail that reside at the clients.
Therefore, the backbone would need to remain inconspicuous
to circumvent SafeSplit. However, in Sect. VI, we demonstrated
that changing the backbone is essential for injecting the
backdoor, such that SafeSplit is also robust against scaling attacks.
Thus, SafeSplit effectively and significantly reduces the
risk of adversaries injecting backdoors into the model while
marginally impacting the benign main task accuracy, fulfilling
our functional and security objectives and requirements.
VIII. RELATED WORK
This section provides an overview of the recent research
progress in the fields of privacy, security, and reliability for
split learning.
Security of Split Learning. As mentioned in Sect. I, split
learning has emerged as a promising alternative to Federated
Learning, offering substantial reductions in the computational
resources required by participants [19], [47], [51]. However,
the inherent data and model control separation in split learning
has raised various security concerns that can be categorized
as the vulnerability to: data reconstruction [38], [13], [18],
[65], label inference attacks [29], [15], [32], [6], and backdoor
attacks [50], [64], [22], [6]. While we only focus on the
backdoor attacks in this paper, we briefly explain the other
attack vectors for completeness.
In a data reconstruction attack, the adversary resides on
the server and tries to recover the original training data from
the feature vector uploaded by the clients. These attacks are
further divided into two categories [65]: passive attacks, where
the server does not disrupt the standard training process,
only leveraging the intermediate steps of the training to gain
information about the samples and create an attack model [13],
[18]. Active data reconstruction attacks instead manipulate the
training process to reconstruct the client data [38], [65] and
obtain better results but faces the risk of being more easily
detectable by the client defenses [13], [16].
In vanilla Split Learning, it is assumed that the clients
cannot access the target labels. At the same time, the server
cannot associate each label with specific samples to maintain
privacy. Therefore, label inference attacks involve an adversary
having control of both, the server and some clients to aim and
steal label information of the data missing from the adversary’s
samples [29], [15], [32], [6]. The defense mechanisms
proposed to counteract label inference attacks primarily rely
on random disturbance and differential privacy [62], [31].
Lastly, in the context of Split Learning, the objective of
backdoor attacks shifts towards implanting backdoors into the
connected model. However, the challenge differs depending
on whether the adversary controls the server or the clients. In
the first scenario, the server aims to successfully compromise
the portion of the model that the clients possess and can attain
this using surrogate clients [50] or shadow models [64]. In the
client-side backdoor attack, the adversary aims to compromise
both, the model residing in the server and, as an additional
challenge, the backdoor needs to persist even on the portion
of models possessed by the other victim clients. To achieve
this, the malicious clients can employ auxiliary models [64] or
poison the training by submitting ad-hoc trigger vectors during
13
training [22], [6]. However, we stress that the existing literature
on client-side backdoor attacks was proposed only for the
vanilla Split Learning framework, where it is assumed that the
clients do not have access to ground truth label information [6].
Hence, mitigating client-side attacks in SL has remained an
open challenge we aim to tackle in this paper for the first
time, especially in the context of U-shaped SL.
Backdoor Defenses in Federated Learning (FL). In the
context of other distributed collaborative learning paradigms,
such as FL the objective of backdoor attacks is to implant
backdoors into the global model. However, the challenge lies
in preserving the efficacy of the local model’s backdoor postaggregation
on the server side. Notable contributions in this
domain include the works of Bagdasaryan et al. [5] and Xie
et al. [60], Saha et al. [43], Shumailov et al. [45], and Wenger
et al. [56], aim to compromise model integrity through subtle
or overt manipulation of the training data.
Multiple defenses have been proposed to mitigate these
attacks in FL [14], [17], [8], [5], [54], [10], [28]. KRUM
calculates the pairwise Euclidean Distances and, analogously
to SafeSplit, sums up the neighborhood to obtain a score.
Afterward, the model with the lowest KRUM score is selected
as the aggregated model. However, the Euclidean distance can
be circumvented as shown by various works for FL [14] and
in Sect. VI for SL. Further, in SL, it is important to select
the latest model rather than the model with the smallest score.
FreqFed [14] also leverages the observation that training a
model for new behavior results in large changes in a model’s
low-frequency components. To detect backdoored models,
clustering is applied to the models’ low frequencies. However,
FreqFed is not applicable to the Split Learning domain due to
the sequentiality of the training steps and other challenges as
mentioned in Sect. III-C. Bagdasaryan et al. [5] consider using
Differential Privacy, but as our evaluation Sect. VI shows,
neither adding random noise to the backbone nor restricting the
L2 − norm of the updates helps in counteracting an advanced
attack in SL. FoolsGold [17] is a FLdefense that assumes high
similarity between model updates submitted by the adversary,
penalizing clients that submit similar (sybils) updates during
aggregation. As we discussed in Sect. I differently from FL,
each client does not start training from the same global model,
but instead, each starts training from a different base model.
Therefore, malicious clients will each impact in different ways
the updates on the backbone. Wang et al. [54] propose a
model poisoning defense that analyzes the latent space of the
second to last layer of the DNN, to detect malicious behavior.
However, as we describe in Sect. III-A in the U-shaped SL
framework, the server does not have access to the last layers,
which are instead controlled by the clients, further as outlined
in Sect. III-C the server should not have access to any training
or validation data. Therefore, it would be unable to analyze
the behavior of the last layers due to its lack of access to
data. Cao et al. introduced a secure aggregation protocol,
where each client is assigned into random subsets of clients,
and for each subset a separate distributed learning process is
executed, resulting in separate models for each subset. Then,
during inference, each global model is queried separately, and
the final label prediction is determined by majority voting.
Unfortunately, repeating the training process and inference for
each sample multiple times results in an impractical overhead.
Furthermore, the defense has been broken for a non-tiny
number of malicious clients. Furthermore, in the case of
highly non-IID data or a limited number of participant clients,
dividing the training into multiple subsets will impact the
overall MA of the trained models.
In comparison, SafeSplit employs different metrics to detect
backdoored models by analyzing models from the static and
dynamic perspectives. The circlewise rollback mechanism
allows skipping detected poisoned models while choosing
the latest benign model to prevent reverting benign training
contributions.
IX. CONCLUSION
In this paper, we proposed SafeSplit, the first defense against
client-side backdoor attacks in SL. Unlike existing methods
for other distributed learning schemes, SafeSplit employs a
rollback mechanism in which we conduct static (frequency)
and dynamic (rotational) analysis to detect poisoned model
updates and limit attack impact. Our extensive evaluation
demonstrated SafeSplit’s effectiveness across various scenarios,
attack settings, and defense-agnostic strategies.
ACKNOWLEDGMENT
This research received funding from the Horizon program of
the European Union under grant agreements No. 101093126
(ACES) and No. 101070537 (CROSSCON), OpenS3 lab, as
well as the Federal Ministry of Education and Research of
Germany (BMBF) within the IoTGuard project and Athene
projects.
REFERENCES
[1] Health Insurance Portability and Accountability Act, 1996.
https://www.govinfo.gov/content/pkg/PLAW-104publ191/pdf/
PLAW-104publ191.pdf.
[2] General Data Protection Regulation, 2018. https://eur-lex.europa.eu/eli/
reg/2016/679/oj.
[3] Pytorch, 2022. https://pytorch.org.
[4] Nasir Ahmed, T Natarajan, and Kamisetty R Rao. Discrete cosine
transform. IEEE transactions on Computers, 100(1):90–93, 1974.
[5] Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and
Vitaly Shmatikov. How To Backdoor Federated Learning. In AISTATS,
2020.
[6] Yijie Bai, Yanjiao Chen, Hanlei Zhang, Wenyuan Xu, Haiqin Weng,
and Dou Goodman. VILLAIN: Backdoor attacks against vertical split
learning. In USENIX Security, 2023.
[7] Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and Seraphin
Calo. Analyzing federated learning through an adversarial lens. In
ICML. PMLR, 2019.
[8] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
Stainer. Machine Learning with Adversaries: Byzantine Tolerant Gradient
Descent. In NIPS, 2017.
[9] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. Fltrust:
Byzantine-robust federated learning via trust bootstrapping. In NDSS,
2021.
[10] Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. Provably secure
federated learning against malicious clients. In AAAI Conference on
Artificial Intelligence, 2021.
14
[11] Jorge Castillo, Phillip Rieger, Hossein Fereidooni, Qian Chen, and
Ahmad Sadeghi. Fledge: Ledger-based federated learning resilient to
inference and backdoor attacks. In ACSAC, 2023.
[12] Wen-Hsiung Chen, C. Harrison Smith, and S. C. Fralick. A fast computational
algorithm for the discrete cosine transform. IEEE Transactions
on communications, 1977.
[13] Ege Erdogan, Alptekin K¨upc¸ ¨u, and A Ercument Cicek. Splitguard:
Detecting and mitigating training-hijacking attacks in split learning. In
Workshop on Privacy in the Electronic Society, 2022.
[14] Hossein Fereidooni, Alessandro Pegoraro, Phillip Rieger, Alexandra
Dmitrienko, and Ahmad-Reza Sadeghi. Freqfed: A frequency analysisbased
approach for mitigating poisoning attacks in federated learning.
In NDSS, 2024.
[15] Chong Fu, Xuhong Zhang, Shouling Ji, Jinyin Chen, Jingzheng Wu,
Shanqing Guo, Jun Zhou, Alex X Liu, and Ting Wang. Label inference
attacks against vertical federated learning. In USENIX Security, 2022.
[16] Jiayun Fu, Xiaojing Ma, Bin B Zhu, Pingyi Hu, Ruixin Zhao, Yaru Jia,
Peng Xu, Hai Jin, and Dongmei Zhang. Focusing on pinocchio’s nose:
A gradients scrutinizer to thwart split-learning hijacking attacks using
intrinsic attributes. In NDSS, 2023.
[17] Clement Fung, Chris JM Yoon, and Ivan Beschastnikh. The limitations
of federated learning in sybil settings. In RAID, 2020.
[18] Xinben Gao and Lan Zhang. PCAT: Functionality and data stealing
from split learning by Pseudo-Client attack. In USENIX Security, 2023.
[19] Yansong Gao, Minki Kim, Sharif Abuadbba, Yeonjae Kim, Chandra
Thapa, Kyuyeon Kim, Seyit A Camtepe, Hyoungshick Kim, and Surya
Nepal. End-to-end evaluation of federated learning and split learning
for internet of things. arXiv preprint arXiv:2003.13376, 2020.
[20] Otkrist Gupta and Ramesh Raskar. Distributed learning of deep neural
network over multiple agents. Journal of Network and Computer
Applications, 116:1–8, 2018.
[21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep
residual learning for image recognition. In IEEE conference on computer
vision and pattern recognition, 2016.
[22] Ying He, Zhili Shen, Jingyu Hua, Qixuan Dong, Jiacheng Niu,Wei Tong,
Xu Huang, Chen Li, and Sheng Zhong. Backdoor attack against split
neural network-based vertical federated learning. IEEE Transactions on
Information Forensics and Security, 2023.
[23] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aur´elien Bellet,
Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles,
Graham Cormode, Rachel Cummings, et al. Advances and open
problems in federated learning. Foundations and trends® in machine
learning, 14(1–2):1–210, 2021.
[24] Sanjay Kariyappa and Moinuddin K Qureshi. Exploit: Extracting private
labels in split learning. In IEEE Conference on Secure and Trustworthy
Machine Learning (SaTML). IEEE, 2023.
[25] Torsten Krauß and Alexandra Dmitrienko. Mesas: Poisoning defense for
federated learning resilient against adaptive attackers. In CCS, 2023.
[26] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of
features from tiny images. Citeseer, 2009.
[27] Kavita Kumari, Phillip Rieger, Hossein Fereidooni, Murtuza Jadliwala,
and Ahmad-Reza Sadeghi. Baybfed: Bayesian backdoor defense for
federated learning. In IEEE S&P. IEEE Computer Society, 2023.
[28] Huimin Li, Phillip Rieger, Shaza Zeitouni, Stjepan Picek, and Ahmad-
Reza Sadeghi. Flairs: Fpga-accelerated inference-resistant & secure
federated learning. arXiv preprint arXiv:2308.00553, 2023.
[29] Oscar Li, Jiankai Sun, Xin Yang, Weihao Gao, Hongyi Zhang, Junyuan
Xie, Virginia Smith, and Chong Wang. Label leakage and protection in
two-party split learning. arXiv preprint arXiv:2102.08504, 2021.
[30] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith.
Federated learning: Challenges, methods, and future directions. IEEE
signal processing magazine, 37(3):50–60, 2020.
[31] Junlin Liu and Xinchen Lyu. Clustering label inference attack against
practical split learning. arXiv preprint arXiv:2203.05222, 2022.
[32] Junlin Liu, Xinchen Lyu, Qimei Cui, and Xiaofeng Tao. Similarity-based
label inference attack against training and inference of split learning.
IEEE Transactions on Information Forensics and Security, 2024.
[33] Yunlong Mao, Zexi Xin, Zhenyu Li, Jue Hong, Qingyou Yang, and
Sheng Zhong. Secure split learning against property inference, data
reconstruction, and feature space hijacking attacks. In European Symposium
on Research in Computer Security. Springer, 2023.
[34] Brendan McMahan and Daniel Ramage. Federated learning: Collaborative
Machine Learning without Centralized Training Data. Google AI,
2017.
[35] H. Brendan McMahan, Daniel Ramage, Kunal Talwar, and Li Zhang.
Learning Differentially Private Language Models Without Losing Accuracy.
In ICLR, 2018.
[36] Luis Mu˜noz-Gonz´alez, Kenneth T. Co, and Emil C. Lupu. Byzantine-
Robust Federated Machine Learning through Adaptive Model Averaging.
In arXiv preprint:1909.05125, 2019.
[37] Ahmed Nasir, Natarajan T, and R Rao Kamisetty. Discrete cosine
transform. IEEE Transactions on Computers, 1974.
[38] Dario Pasquini, Giuseppe Ateniese, and Massimo Bernaschi. Unleashing
the tiger: Inference attacks on split learning. In CCS, 2021.
[39] Maarten G. Poirot, Praneeth Vepakomma, Ken Chang, Jayashree
Kalpathy-Cramer, Rajiv Gupta, and Ramesh Raskar. Split learning
for collaborative deep learning in healthcare. arXiv preprint
arXiv:1912.12115, 2019.
[40] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min
Lin, Fred A. Hamprecht, Yoshua Bengio, and Aaron Courville. On
the spectral bias of neural networks. In International Conference on
Machine Learning, 2019.
[41] Phillip Rieger, Torsten Krauß, Markus Miettinen, Alexandra Dmitrienko,
and Ahmad-Reza Sadeghi. Crowdguard: Federated backdoor detection
in federated learning. In NDSS, 2024.
[42] Phillip Rieger, Alessandro Pegoraro, Kavita Kumari, Tigist Abera,
Jonathan Knauer, and Ahmad-Reza Sadeghi. Safesplit: A novel defense
against client-side backdoor attacks in split learning. In NDSS, 2025.
[43] Aniruddha Saha, Akshayvarun Subramanya, and Hamed Pirsiavash.
Hidden trigger backdoor attacks. In AAAI, 2020.
[44] Shiqi Shen, Shruti Tople, and Prateek Saxena. Auror: Defending Against
Poisoning Attacks in Collaborative Deep Learning Systems. In ACSAC,
2016.
[45] Ilia Shumailov, Zakhar Shumaylov, Dmitry Kazhdan, Yiren Zhao, Nicolas
Papernot, Murat A Erdogdu, and Ross J Anderson. Manipulating sgd
with data ordering attacks. NeurIPS, 2021.
[46] K Simonyan and A Zisserman. Very deep convolutional networks for
large-scale image recognition. In ICLR. Computational and Biological
Learning Society, 2015.
[47] Abhishek Singh, Praneeth Vepakomma, Otkrist Gupta, and Ramesh
Raskar. Detailed comparison of communication efficiency of split
learning and federated learning. arXiv preprint arXiv:1909.09145, 2019.
[48] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel.
The german traffic sign recognition benchmark: a multi-class classification
competition. In International joint conference on neural networks.
IEEE, 2011.
[49] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew
Rabinovich. Going deeper with convolutions. In IEEE conference on
computer vision and pattern recognition, 2015.
[50] Behrad Tajalli, O˘guzhan Ersoy, and Stjepan Picek. On feasibility of
server-side backdoor attacks on split learning. In IEEE Security and
Privacy Workshops (SPW). IEEE, 2023.
[51] Chandra Thapa, Mahawaga Arachchige Pathum Chamikara, and Seyit A
Camtepe. Advancements of federated learning towards privacy preservation:
from federated learning to split learning. Federated Learning
Systems: Towards Next-Generation AI, pages 79–109, 2021.
[52] Daniel J Velleman. How to prove it: A structured approach. Cambridge
University Press, 2019.
[53] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and Ramesh
Raskar. Split learning for health: Distributed deep learning without
sharing raw patient data. arXiv preprint arXiv:1812.00564, 2018.
[54] Ning Wang, Yang Xiao, Yimin Chen, Yang Hu, Wenjing Lou, and
Y Thomas Hou. Flare: defending federated learning against model
poisoning attacks via latent space representations. In Asia Conference
on Computer and Communications Security, 2022.
[55] Zhongde Wang. Fast algorithms for the discrete w transform and for
the discrete fourier transform. IEEE Transactions on Acoustics, Speech,
and Signal Processing, 1984.
[56] Emily Wenger, Josephine Passananti, Arjun Nitin Bhagoji, Yuanshun
Yao, Haitao Zheng, and Ben Y Zhao. Backdoor attacks against deep
learning systems in the physical world. In IEEE conference on computer
vision and pattern recognition, 2021.
[57] Alexander Wong, Mohammad Javad Shafiee, and Michael St Jules.
Micronnet: A highly compact deep convolutional neural network architecture
for real-time embedded traffic sign classification. IEEE Access,
6:59803–59810, 2018.
15
[58] Xiao, Wu Xiaolin, and Liu Bolin. A study on quantization effects of dct
based compression. IEEE International Conference on Image Processing
(ICIP), 2017.
[59] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel
image dataset for benchmarking machine learning algorithms, 2017.
[60] Chulin Xie, Keli Huang, Pin-Yu Chen, and Bo Li. DBA: Distributed
backdoor attacks against federated learning. In ICLR, 2020.
[61] Zhi-Qin John Xu, Yaoyu Zhang, and Yanyang Xiao. Training behavior of
deep neural network in frequency domain. In International Conference
on Neural Information Processing. Springer, 2019.
[62] Xin Yang, Jiankai Sun, Yuanshun Yao, Junyuan Xie, and Chong Wang.
Differentially private label protection in split learning. arXiv preprint
arXiv:2203.02073, 2022.
[63] Ziyuan Yang, Yingyu Chen, Huijie Huangfu, Maosong Ran, Hui Wang,
Xiaoxiao Li, and Yi Zhang. Robust split federated learning for u-shaped
medical image networks. arXiv preprint arXiv:2212.06378, 2022.
[64] Fangchao Yu, Lina Wang, Bo Zeng, Kai Zhao, Zhi Pang, and Tian Wu.
How to backdoor split learning. Neural Networks, 168:326–336, 2023.
[65] Fangchao Yu, Bo Zeng, Kai Zhao, Zhi Pang, and Lina Wang. Chronic
poisoning: Backdoor attack against split learning. In AAAI, 2024.
[66] Sergey Zagoruyko. Wide residual networks. arXiv preprint
arXiv:1605.07146, 2016.
[67] Chenchen Zhou, Hongbo Cao, Yingying Zhao, Sicong Zhao, and Yongqi
Sun. Label inference attack based on soft label towards two-party split
learning. In IEEE International Symposium on Parallel Architectures,
Algorithms and Programming (PAAP). IEEE, 2023.
[68] Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui
Xiao, and Beng Chin Ooi. Passive inference attacks on split learning
via adversarial regularization. arXiv preprint arXiv:2310.10483, 2023.
APPENDIX
A. Deep Neural Network (DNN)
A DNN is a mathematical function denoted as F(X;W),
with X representing the input data samples and W denoting
the network’s parameters (comprising weights and biases).
This network is structured into several layers, denoted as Fi,
where i ∈ {1, · · · ,L}. The first layer, the input layer, is
labeled as F1, while the final layer, termed the output layer,
is designated as FL. The intermediate layers are commonly
referred to as hidden layers. In a feed-forward neural network,
data moves in a unidirectional path, starting from the input
layer, traversing through the hidden layers, and ultimately
reaching the output layer.
B. Poisoning Attack
In machine learning security, poisoning attacks represent a
scheme in which model parameters are intentionally manipulated
during or after training to introduce abnormal behaviors.
Targeted or backdoor attacks alter the DNN’s training
stealthily to generate specific mispredictions when the model
is presented with inputs containing predetermined triggers. For
example, a trigger could be a red pixel placed in the upper left
corner of an input image, which must be incorporated into
the training dataset through data poisoning, and the abnormal
behavior would be the classification of all samples with the
red pixel into a specific predetermined class. The success rate
for the backdoor attack is calculated based on the prediction
performance on triggered data, denoted as Backdoor Accuracy
(BA), while maintaining the expected behavior on benign
inputs, as indicated by a high model or main-task accuracy
(MA).
Following Fig. 1, we can see how an adversary must
manipulate one or more clients to execute backdoor attacks
in the Split Learning configuration. The adversary trains the
poisoned local head and tail portion to poison the backbone
in the central server and the subsequent head and tail training
of the future clients. The critical objective is to poison the Ushaped
aggregated model’s prediction without incurring any
noticeable behavior change to the server backbone and the
other clients’ head and tail, thus undermining the integrity
and reliability of the entire system [22], [6].
C. Discrete Cosine Transform
In signal processing, the Discrete Cosine Transform
(DCT) [37] is employed to break down a signal into its
frequency components, providing insights into the underlying
dynamics and transitions within the signal[58]. The DCT takes
a sequence of numbers and represents them as a combination
of simple wave patterns (sinusoids) with different frequencies
and sizes.
Mathematically, DCT transformations are invertible functions
that convert an input sequence of N real numbers into the
coefficients (values that multiply the wave patterns) of N orthogonal
cosine basis functions (wave patterns independent of
each other) with increasing frequencies. The DCT components
(the coefficients) are ordered by significance (importance),
with the first coefficient representing the sum of the input
sequence normalized by length. Lower-order coefficients correspond
to lower signal frequencies (slower repeating waves)
and indicate the sequence’s patterns (general trends or shapes
in the data). These are known as low-frequency components.
The 2-D DCT of a signal S (e.g., a matrix of size N by M)
at frequencies k and l (X(k, l)), is given by the following
equation [12], [55], [4], [14].
X(k, l) = akal
MX−1
m=0
NX−1
n=0
S(m, n)cos

kπ
2M
(2m+1)

cos

lπ
2N
(2n+1)

(8)
Where ak, al are dependent on the values of k, and l, with
the following rules:
ak =
(q
2
MN for k = 0
1 for k = 1, 2, ...,M − 1
(9)
al =
(q
2
MN for l = 0
1 for l = 1, 2, ...,N − 1
(10)
As we elaborate in Sect. V-C, recent work showed that
in the early stages of the training, mostly the low-frequency
components change while the high-frequency components
change during the fine-tuning, when the model is already close
to convergence [40], [61]. In Sect. V-C, we describe how this
can be used to help detect models with injected backdoor
behavior.
D. Effectiveness of the Angular Distance
SafeSplit’s selection mechanism relies on the Euclidean
distance in the frequency domain and the rotational distance
among server model states. SafeSplit uses the assumption that
the majority of clients are benign. We establish a norm by
accepting a set of N/2+1 clients with the least Euclidean and
16
0 100 200 300 400 500
Training Step
1
2
3
Angular Distance
max(Majority)
min(Majority)
Last Benign
Last Poisoned
(a) Rotational Distance Scores
0 100 200 300 400 500
Training Step
0
200
400
Euclidean Distance
max(Majority)
min(Majority)
Last Benign
Last Poisoned
(b) Frequency Distance Scores
Fig. 8: Rotational and Frequency distance scores for an IIDrate
of 1.0.
rotational distances. Fig. 8 illustrates a distinct gap between
malicious and benign scores for an IID-rate of 1.0 and Fig. 9.
The figures visualize how the different perspectives (static and
dynamic) complete it each other. As long as malicious scores
exceed the maximum score of the majority set for at least
one of both metrics, a malicious client remains unselected,
preventing any backdoor injection. Only occasionally, a benign
score may surpass the maximum of the majority set, resulting
in its exclusion. This exclusion can impact MA, albeit
minimally, as demonstrated in Table II. Our evaluation reveals
that the score gap between benign and malicious instances is
more pronounced during the model’s learning phase, gradually
diminishing as MA converges. Hence, halting training at MA
convergence proves advantageous.
E. Impact of Misclassifications on BA
For pixel-trigger backdoors the BA is often not 0%, even
when the model is benign and not poisoned. As outlined by
Fereidooni et al., this phenomenon occurs due to the misclassification
of samples by the model, if the MA is not 100%.
Especially, samples that are similar to the backdoor target
label are, if not recognized correctly, likely to be incorrectly
classified as the backdoor target, even when they are independent
of the presence of the backdoor trigger. For example,
in the case of truck images, the model can misclassify them
as car images. If a triggered sample is misclassified as the
backdoor target by coincidence, it is counted as successful
0 100 200 300 400 500
Training Step
1
2
3
Angular Distance
max(Majority)
min(Majority)
Last Benign
Last Poisoned
(a) Rotational Distance Scores
0 100 200 300 400 500
Training Step
0
500
1000
1500
Euclidean Distance
max(Majority)
min(Majority)
Last Benign
Last Poisoned
(b) Frequency Distance Scores
Fig. 9: Rotational and Frequency distance scores for an IIDrate
of 0.8.
backdoor activation and increases the BA, although the model
was trained to recognize the backdoor trigger [14].
To illustrate this, we show in Fig. 10 a confusion matrix for
a poisoned test dataset evaluated on a benign model. In this
case, the backdoor trigger, represented by a red rectangle, is
intended to cause the model to predict class 2 (bird).
As the figure shows, although the model does not contain
any backdoor, in 946 cases, label 2 is predicted, resulting in
a BA of 10.5%.
Notably, the red rectangle also covers parts of the image,
thus affecting the model’s ability to recognize the actual object.
F. Evaluation of Further Attacks
In addition, we evaluated SafeSplit for a label swapping
attack, where the predictions for all samples of two classes
shall be swapped, thus realizing a mixture of backdoor and
untargeted attack. For this experiment, we measured the effectiveness
using the attack success rate (ASR), counting the
fraction of samples belonging to both classes where the model
predicts the swapped labels. We performed this experiment for
all pairs of classes using the CIFAR-10 dataset. We observed
that SafeSplit effectively reduces the ASR in average from
22.7% to 3.8%. Notably, the ASR for the undefended model
(22.7%) is significantly smaller than for regular backdoor
attacks such as the semantic backdoor. This is caused by the
clean samples that are part of the benign clients’ datasets,
allowing them to reduce the attack’s impact, while for backdoors,
such as the semantic backdoor, benign clients pose only
17
a negligible number of samples. Further, similar to the pixelpattern
backdoor, the BA for a benign model is not exactly
0% but 3.4% (cf. Sect. E).
0 1 2 3 4 5 6 7 8 9
Predicted Label
9 8 7 6 5 4 3 2 1 0
True Label
377 227 217 2 151 8 8 0 7 3
1 972 11 1 7 1 5 0 0 2
0 0 0 0 0 0 0 0 0 0
8 83 161 79 340 290 37 0 0 2
6 31 86 1 847 16 12 0 0 1
5 38 112 13 216 603 8 2 0 3
2 76 151 4 183 35 549 0 0 0
16 85 131 4 528 80 5 149 0 2
81 539 55 0 75 6 5 0 235 4
13 677 22 1 15 12 3 0 0 257
Confusion Matrix
0
200
400
600
800
1000
Fig. 10: Confusion matrix for triggered inputs applied on a
benign model.
While also untargeted attacks fall outside our threat model’s
scope, since they can be straightforwardly detected and inherently
differ from backdoor attacks, we tested a lossmaximization
attack and showed that SafeSplit mitigates the
attack and maintains the Main Task Accuracy (MA) at 64%.
G. Runtime Evaluation
We evaluated the runtime performance of SafeSplit and its
individual components in dependence of the client number to
determine its scalability. Since the defense is executed before
every training, we performed for each client number an experiment
running 50 rounds and measured the individual runtimes
every time the defense was called. Thus, depending on the
client number, we obtained measurements of 245 (5 clients)
and 1470 (30 clients). Notably, we omitted measurements from
the first round until every client provided at least one model.
We trimmed the 5% highest and smallest values and averaged
the remaining values. The results are shown in Fig. 11. As
the figure shows, the runtime of SafeSplit scales linearly as
the most time-consuming operations, such as the frequency
transformation, are executed once per model. Notably, the
runtime for the frequency transformation is plotted as an individual
curve but also included in the curve for the Frequency
Analysis.
5 10 15 20 25 30
Num. Clients
0
20
40
60
80
Duration (s)
Rotational Metric
Frequ. Transformation
Frequ. Metric
Total
Fig. 11: Evaluation times of SafeSplit and its individual
components for different client numbers.
Notably, all code was written in Python and no parallelization
was used, although operations such as the frequency
transformation can be easily parallelized. While runtime engineering
is out of the scope of this work, this demonstrates
the practical applicability of SafeSplit.
H. Details on Rotational Distance
The rotational distance metric is used to analyze the directional
changes in the backbone’s parameter space over training
rounds. It provides a way to measure how the configuration of
the model’s parameters evolves dynamically over time. Unlike
traditional magnitude-based metrics, which focus only on the
size of updates, rotational distance captures the trajectory and
orientation shifts of model updates, making it more robust
against adversarial manipulations.
Parameter Representation and Transformation: The backbone
of the model consists of high-dimensional weight tensors.
Directly analyzing these tensors would be inefficient and
difficult to interpret in terms of rotation. Instead, the rotational
metric first extracts the backbone parameters and reshapes
them into a structured representation that allows for spatial
and directional analysis. This is done by first transforming
the weight tensors into a 2D matrix form and then computing
coordinate-based transformations (i.e., mapping into separate
x- and y-coordinate vectors).
To create a coordinate space for the weights, we compute
mean values along the rows and columns of the transformed
weight tensors to obtain one-dimensional vectors of the mean
values of rows and columns. Next, these obtained vectors are
multiplied with the 2D matrix to preserve the variability of the
values in the original backbone tensor. Finally, the obtained
two 2D matrices are flattened to construct the x- and ycoordinate
vectors Bx
t and By
t . This transformation allows the
weight tensors to be entangled into x and y coordinate tensors,
which will be used for computing angular displacement, as
detailed in the following.
Angular Displacement Computation: Once the weight parameters
are mapped into the coordinate space, the next
step is to measure their angular displacement. The angular
displacement θ(t) is defined as the coordinate-wise rotation
of each paired x- and y coordinate, obtained from Bx
t and By
t
(i.e. rotation from the x-axis). This is computed as:
θ(t) = arctan (Bx
t ,By
t )
This formulation ensures that small shifts in weight direction
are captured, even if the magnitude of the update remains
the same. We apply a geometric transformation to the mapped
weight tensors and compute the pairwise angular difference
between successive updates. The use of arctan in this context
is more than just a cosine similarity measure. It leverages the
transformed x- and y-coordinate vectors to compute the angle
between two vectors in a way that accurately captures the
global directionality of parameter updates.
arctan is used to compute the orientation of the x- and
y-coordinate vectors relative to the origin. This approach
18
provides an equivalent measure of angular displacement but is
expressed in direct coordinate-wise rotation (from the x-axis)
rather than the computation of backbone similarity by utilizing
the final x- and y-coordinate vectors. arctan determines the
relative rotation of each coordinate point, effectively capturing
local changes in parameter orientation.
Estimating Rotational Frequency: After computing the angular
displacement, the rotational metric needs to determine
how frequently these shifts occur over time. This is done
by measuring the rate of angular displacement per unit time,
which corresponds to angular velocity:
ω(t) =
θ(t) − θ(t − 1)
Δt
where Δt represents the time interval between consecutive
updates. It is set to 1 in our approach. This provides an estimate
of how quickly the orientation of the model parameters
is changing. Finally, we convert the obtained angular velocity
into rotational frequency, normalized by the full rotation cycle:
RD =
ω(t)
2π
This transformation ensures that the rotational distance is
expressed in a form that captures repetitive shifts in parameter
orientation over multiple training rounds.
Pairwise Comparison of Rotational Frequencies: Next,
we compare the rotational frequencies across different models.
The rotational distance metric RD is designed to track
deviations from expected training trajectories by analyzing
how different clients’ backbones behave over training rounds.
Instead of comparing updates for a single model over time, it
measures differences in rotational frequencies across multiple
clients. The final rotational distance score Ri is obtained by
summing the absolute differences:
Ri =
X
|RDi − RDj |
Hence, RD allows the detection of anomalous client behavior,
as adversarially manipulated models tend to exhibit higher
frequency deviations compared to benign training updates.
19