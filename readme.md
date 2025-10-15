# GNN-MA: Graph Matching Network for 3D Molecular Similarity & Virtual Screening

GNN-MA is a graph neural network framework for 3D ligand similarity alignment and virtual screening. It supports cross - graph attention, node - edge collaborative convolution, and evaluation on multiple datasets (DUD - E, LIT - PCBA/PCBA). It also provides complete data pre - processing, training, and evaluation scripts. 
For the complete code, please refer to: https://github.com/BobbyLiukeling/GNN - MA

## 1. Environmental Requirements

OS：Linux / macOS / Windows(Linux is recommended) are all acceptable.

Python：3.10

Dependency Installation (Conda Recommended)

1. numpy                             1.23.5
2. pandas                            2.2.3
3. rdkit                             2025.3.4
4. torch                             2.5.1
5. matplotlib                        3.7.2

## 2. Data Download and Pre - processing
This model adopts the virtual screening benchmark datasets DUD - E and LIT - PCBA, and the data will be processed into NPZ format before training.
1. Download from the official website of the virtual screening benchmark dataset DUD - E (https://dude.docking.org/) and the website of LIT - PCBA (http://drugdesign.unistra.fr/LIT - PCBA) respectively, and store them in the DUD - E and LIT - PCBA folders. Due to storage space limitations, there is only one protein target in both DUD - E and LIT - PCBA. You can download the complete data and place it in the specified folder for training.
2. Process the DUD - E data, split the merged decoy molecules, use split_DUDE.py for processing, and store the processed data in the DUD - E - dealed folder.
3. Encode the molecules in the DUD - E and LIT - PCBA datasets, encode different types of molecular data into npz format, use encoding.py and encoding - LIT.py for processing respectively, and store them in the encode - DUE - E and encode - LIT folders after processing.
4. All data - related processing codes and files are stored in the data folder.

## 3. GNN-MA Model Construction
GNN-MA is a graph neural network model designed for three-dimensional molecular similarity alignment. The model is based on a graph convolutional neural network, augmented with a cross-graph attention mechanism to capture inter-molecular dependencies, enabling the molecular of atomic, bond, and pharmacophore features for molecular alignment and similarity evaluation.
GNN-MA supports molecular comparison and alignment at multiple dimension, making it broadly applicable to structure-based drug design tasks.
In this study, molecular structures are represented as undirected weighted graphs , where the graph node set H corresponds to atoms and the edge set  corresponds to chemical bonds. The connectivity between atoms is encoded by an adjacency matrix , where  denotes the number of atoms in a molecule. Training samples are provided as molecular pairs, meaning that the model simultaneously receives two molecular graphs,  and , as input. Each molecular is described by three core components: an atom(node) feature matrix, a bond(edge) feature tensor, and an adjacency matrix. We adopt one-hot encoding to uniformly represent all feature. Based on extensive prior studies and our comparative experiments, we selected the following descriptors to represent atomic and bond features.

The specific content is implemented in GNN-MA.py.

<img src="Figure 1.png" alt="Alternative Text" width="1168" height="458">

### 3.2.1 Intra-Graph Message Passing

Within a single molecule, atom and bond are updated simultaneously through a cooperative convolution process (Figure 1b).


**3.2.1 Intra-Graph Message Passing**

Within a single molecule, atom and bond are updated simultaneously through a cooperative convolution process, as illustrated in Figure 1(b).

(1) Neighborhood aggregation for atoms and bonds follows eqs 1 and 2.
$$
l(h)=\varphi(AH\,W_h)                                    \tag{1}
$$

$$
l(e)=\varphi(AE\,W_e)                                    \tag{2}
$$

Where, $W_h$ and $W_e$ are learnable weight parameters for atom and bond, respectively. $\ell^{(h)}$ and $\ell^{(e)}$ denote the aggregated atom and bond information, and $\varphi$ represents the activation function.

(2) Fusion of node and edge information

Node fusion is defined in eq 3				
$$
\widetilde{H}=\mathrm{Dropout}\,\varphi\big(\ell^{(h)}+\ell^{(e)}\big) \tag{3}
$$

Edge fusion is defined in eq 4
$$
\widetilde{E}=f_{\text{mask}}\big(A_{ij},\ f_{\text{edge}}(\widetilde{H}_i,\widetilde{H}_j,e_{ij})\big)\big) \tag{4}
$$
Here, $\widetilde{H}_i$ and $\widetilde{H}_j$ denote the information of the i-th and j-th atoms, respectively. The function $f_{\text{edge}}$ fuses the bond feature $e_{ij}$ with the updated features of atom and . The function $f_{\text{mask}}$ applies a linear transformation to the bond representation using the adjacency matrix.



### 3.2.2 Cross-graph attention
Given two molecular graphs $G_1$ and $G_2$, the representation of $G_1$ is updated by integrating information from $G_2$, thereby enabling $G_1$ to capture cross-graph dependencies, as illustrated in Figure 1(c).

1.  Computation of atoms attention scores is defined in eq 5.
$$
Q_{G_1}^{(h)}=\widetilde{H}_{G_1}W_q^{(h)},\ \ K_{G_2}^{(h)}=\widetilde{H}_{G_2}W_k^{(h)},\ \ V_{G_2}^{(h)}=\widetilde{H}_{G_2}W_v^{(h)} \tag{5}
$$


Where $W_q^{(h)}$, $W_k^{(h)}$, $W_v^{(h)}$ and are learnable weight matrices, while $\widetilde{H}_{G_1}$ and $\widetilde{H}_{G_2}$ denote the atom features of $G_1$ and $G_2$ after intra molecular graph convolution.

The atoms information in $G_2$ is then aggregated into the atoms of $G_1$ according to eq 6:
$$
\overleftrightarrow{H}_{G_1 \leftarrow G_2}
= \operatorname{softmax}\!\left(
  \frac{\,Q_{G_1}^{(h)} \big(K_{G_2}^{(h)}\big)^{\top}}{\sqrt{d_h}}
\right) V_{G_2}^{(h)} \tag{6}
$$




2. Computation of bonds attention scores is defined in eq 7.
$$
Q^{(e)}_{G_1}=\widetilde{E}_{G_1}W^{(e)}_{q},\quad
K^{(e)}_{G_2}=\widetilde{E}_{G_2}W^{(e)}_{k},\quad
V^{(e)}_{G_2}=\widetilde{E}_{G_2}W^{(e)}_{v} \tag{7}
$$

Here, $W^{(e)}_{q}$ $W^{(e)}_{k}$ and $W^{(e)}_{v}$ are learnable weight matrices, while $\widetilde{E}_{G_1}$ and $\widetilde{E}_{G_2}$ denote the bond features of $G_1$ and $G_2$ after intra molecular graph convolution.

The bonds information from $G_2$ is then aggregated into the bonds of $G_1$ according to eq 8:
$$
\overleftrightarrow{E}_{G_1 \leftarrow G_2}
= \operatorname{softmax}\!\left(
  \frac{\,Q^{(e)}_{G_1}\big(K^{(e)}_{G_2}\big)^{\top}}{\sqrt{d_h}}
\right)\, V^{(e)}_{G_2} \tag{8}
$$



### 3.2.3 bond to atom Aggregation

Within each molecule, the updated bond information through cross molecular graph attention is aggregated to the corresponding atoms.

First, for a given atom $i$, the information from all bonds connected to it is aggregated as follow eqs 9 and 10:
$$
				m_i=\sum_{j=1}^{N} A_{ij}(\overleftrightarrow{E}_{ij} + \overleftrightarrow{E}_{ij}) \tag{9}
$$

$$
				M=\begin{bmatrix} m_1 \\ \vdots \\ m_N \end{bmatrix} \in \mathbb{R}^{N\times h} \tag{10}
$$

Here,$\overleftrightarrow{E}_{ij}$ represents the bond between atoms $i$ and $j$ , which are refined through graph convolution and cross graph attention.

Finally, aggregated all atom information is defined in eq 11:
$$
				H^{(e\to n)}=\varphi(MW) \tag{11}
$$


where $w$ is a trainable weight matrix.





### **3.2.4 Data fusion and get similarity scoring**

After intra graph convolution, cross graph attention, and bond to atom message passing, the updated molecular signal are integrated through residual aggregation is defined in eq 12, and as illustrated in Figure 1(d):

$$
				G=\widetilde{H}+\overleftrightarrow{H}+H^{(e\to n)} \tag{12}
$$

For similarity evaluation, the two molecular graphs $G_1$ and $G_2$ are concatenated to form a joint representation is defined in eq 13:
$$
				z=[G1\,||\,G2] \tag{13}
$$
where "$||$" denotes the feature concatenation operator. The similarity score is then computed is defined in eq 14:
$$
				s=w_2^{\top}\, \mathrm{Dropout}\!\big( \varphi(W_1 z + b_1) \big) + b_2 \tag{14}
$$


Here, $W_2,W_1, b_1$ and $b_2$ are learnable parameters, and represents the final similarity score.





## train
The model was trained in a supervised learning framework, Each training sample was formed by combining a ligand with either an active or a decoy compound. Binary cross-entropy loss was employed as the loss function , and model parameters were optimized using the Adam algorithm. The initial learning rate was set to ,batch size is 32, and training proceeded for 20 epochs.

Since the DUD - E and LIT - PCAB datasets are organized in different ways, the codes used during training also have some differences. Among them, train.py is the code for training the model on the DUD - E dataset, and train - LIT.py is the code for training the model on the LIT - PCBA dataset. A GPU can be used during training, and a CPU can be used as a substitute on devices without a GPU.



