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

![GMNS-flow-1-21](D:\西华\paper\GNN-MA\2025-11\img\GMNS-flow-1-21.png)



Figure 1. Overall framework of GNN-MA

As illustrated in Fig. 1(A)–(D), GNN-MA tackles ligand-based virtual screening (LBVS) as a query–candidate pairwise scoring task. Given a query molecule ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps3.jpg) and a candidate molecule ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps4.jpg) , the model first encodes node- and edge-level features of their molecular graphs (Fig. 1A) and learns structural representations via message passing within each graph (Fig. 1B). It then applies cross-graph attention to explicitly capture atom-level and bond-level interactions between the two molecules, producing an interpretable soft alignment (Fig. 1C). Finally, the interaction-aware representations are fused and pooled to yield a matching score for ranking (Fig. 1D).

This framework places no restriction on input dimensionality: 1D/2D information as the default set of features, while 3D-related cues are incorporated into the same unified pipeline as optional enhancements when available. To meet the demands of large-scale screening, we further benchmark inference throughput in the experimental section to characterize the scalability of the proposed approach.

***\*3.1 problem define\****

In this study, we formulate LBVS as a pairwise scoring problem between a query molecule ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps5.jpg) and a candidate molecule ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps6.jpg). The model outputs a matching score ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps7.jpg), based on which the candidate set is ranked to achieve early enrichment. During training, we use labeled molecular pairs ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps8.jpg), where ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps9.jpg).

***\*3.2 Molecular Graph Representation and Input Features\****

We represent each molecule as a graph ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps10.jpg), where ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps11.jpg) is the atom feature matrix, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps12.jpg) is the number of atoms, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps13.jpg) is the node-feature dimension, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps14.jpg) is the bond feature tensor, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps15.jpg) is the edge-feature dimension. To support practical LBVS with heterogeneous input availability, we adopt a dimension-agnostic design. The model uses 2D topology with atom/bond attributes by default, and integrates 3D information as an optional augmentation when present.  Tables 1 and 2 summarize the feature categories.



Table 1. Atom features

| **No.** | **Feature category**    | **Meaning / examples**                                       | **Notes (task relevance)**                                   |
| ------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1       | Atom type               | C, O, N, S, etc.                                             | Basic chemical composition that influences molecular properties and interaction patterns. |
| 2       | Topological position    | On the main scaffold / on a side chain                       | Distinguishes the structural core from substituents, affecting conformational flexibility and local interactions. |
| 3       | Scaffold type           | Aliphatic scaffold, aromatic scaffold, etc.                  | Reflects global structural framework differences, often associated with hydrophobicity, rigidity, and overall molecular shape. |
| 4       | Aromaticity             | Whether the atom belongs to an aromatic ring                 | Aromaticity affects electron distribution and pi-pi interactions, which are frequently linked to bioactivity. |
| 5       | Ring membership         | Whether the atom is part of any ring system                  | Ring membership impacts rigidity, geometry, and accessibility, thereby influencing structural matching in screening. |
| 6       | Pharmacophoric features | H-bond donor, H-bond acceptor, aromatic ring, hydrophobic site, etc. | Captures key structural motifs closely related to ligand-target interactions and biological activity. |

Table 2. Bond features

| **No.** | **Feature category** | **Meaning / examples**            | **Notes (task relevance)**                                   |
| ------- | -------------------- | --------------------------------- | ------------------------------------------------------------ |
| 1       | Bond type            | Single, double, triple, aromatic  | Determines connectivity strength and geometric/electronic structure; fundamental to molecular topology and reactivity. |
| 2       | Aromatic bond        | Whether the bond is aromatic      | Indicates conjugated/pi systems, affecting electron delocalization and molecular recognition (e.g., pi-pi interactions). |
| 3       | Conjugation          | Whether the bond is conjugated    | Related to electron delocalization, influencing polarity, stability, and interaction patterns. |
| 4       | In-ring bond         | Whether the bond is within a ring | Ring bonds constrain molecular geometry and rigidity, affecting structural matching during screening. |

***\*3.3 Intra-molecular Message Passing\****

To learn intra-molecular structural semantics, GNN-MA performs ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps16.jpg)-th layer message passing on the query graph ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps17.jpg) and the candidate graph ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps18.jpg) separately. Given the (![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps19.jpg)-1)th layer node embedding ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps20.jpg) and the edge (bond) representation ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps21.jpg) between nodes ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps22.jpg) and ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps23.jpg), node ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps24.jpg) aggregates messages from its neighborhood ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps25.jpg) as follows:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps26.jpg) 

The node embeddings are then updated via fallow function:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps27.jpg) 

Here, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps28.jpg) and ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps29.jpg) are learnable mappings (e.g., multilayer perceptrons), that integrate neighborhood atom and bond information into the node representation. To improve training stability and representation capacity, we further apply residual connections and normalization between layers.

***\*3.4 cross-graph attention and soft correspondences\****

Independent encoding of each molecule is typically insufficient to extract the pairwise matching cues required for retrieval. Therefore, GNN-MA builds upon the intra-molecular representations with a cross-graph attention module that explicitly captures atom and bond level interactions between the query and candidate, producing an interpretable soft-alignment matrix.

Let ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps30.jpg)and ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps31.jpg) denote the atom representations of the two compounds after intra-molecular message passing, where ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps32.jpg) and ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps33.jpg) are the numbers of atoms, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps34.jpg) is the embedding dimension. We use scaled dot-product attention to compute the cross-graph relevance between the ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps35.jpg)-th atom in the query molecule ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps36.jpg) and the ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps37.jpg) atom in the candidate molecule ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps38.jpg) :

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps39.jpg) 

Here, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps40.jpg) and ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps41.jpg) are learnable parameters, and ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps42.jpg) is the scaling factor used in the attention mechanism.

For each query atom ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps43.jpg), we normalize over all candidate atoms ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps44.jpg) by applying a softmax along the candidate-atom dimension, yielding the soft alignment weights from query to candidate:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps45.jpg) 

Based on these weights, query atom ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps46.jpg) aggregates information from the candidate molecule to obtain a cross-graph contextual representation:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps47.jpg) 

To enhance the symmetry of alignment and the complementarity of information, we similarly compute the reverse attention ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps48.jpg) and obtain ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps49.jpg). 

Analogously, we also construct cross-graph attention at the bond-level: the correlation computation, normalization, and cross-graph aggregation follow the same procedure as the atom-level attention, except that the inputs are replaced from atom representations to bond representations.

**3.5 Edge Fusion and Bond-to-Atom Aggregation**

The atom-level interaction representations produced by cross-graph attention provide evidence of *soft correspondences* between the query and candidate. Building on this, GNN-MA introduces a two-stage structural enhancement prior to node updating, namely edge-level fusion → edge-to-node aggregation: we first fuse information at the bond (edge) level to obtain enhanced edge representations, and then aggregate these enhanced edge features to adjacent atoms. In this way, bond semantics are injected into atom representations and subsequently contribute to the final scoring.

***\*（1）\*******\*Edge-level Fusion:\****

For each chemical bond, we construct a fused edge representation by combining the cross-graph–updated representations of its two incident atoms with the original bond feature:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps50.jpg) 

Here, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps51.jpg) is a learnable mapping that integrates node semantics and bond information to produce an enhanced bond representation.

***\*（2）\*******\*Edge to node aggregation\****

After obtaining the fused edge representations, we aggregate information from incident edges for each atom to form an edge-aggregation vector:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps52.jpg) 

Then, we inject the aggregated bond information into the atom representation through a learnable transformation and an update operation:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps53.jpg) 

Here, ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps54.jpg) is a learnable mapping that transfers bond-level information to atom-level representations, facilitating subsequent fusion with atom features.

***\*3.6 Similarity Scoring and Ranking\****

***\*Graph-level pooling:\**** we fuse the molecular representations obtained from intra-graph convolution, cross-graph attention, and edge-to-node aggregation via a residual combination, and then apply a readout function to obtain graph-level embeddings for scoring:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps55.jpg) 

***\*Pairwise scoring:\**** we combine the two graph-level embeddings and feed them into a multilayer perceptron to obtain the final matching score:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps56.jpg) 

Here, || denotes feature concatenation.

First, the model outputs a matching score ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps57.jpg) for each query–candidate molecular pair. During training, we use the binary cross-entropy loss ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps58.jpg) as the primary supervision signal for active/decoy classification. In addition, we introduce a within-target pairwise ranking constraint ![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps59.jpg): from the negative set, we select the Top-K hardest negatives (hard negatives) and explicitly enforce their scores to be lower than those of positive samples, thereby strengthening early enrichment. The overall objective is finally defined as:

![img](file:///C:\Users\11943\AppData\Local\Temp\ksohtml16352\wps60.jpg)





## train
The model was trained in a supervised learning framework, Each training sample was formed by combining a ligand with either an active or a decoy compound. Binary cross-entropy loss was employed as the loss function , and model parameters were optimized using the Adam algorithm. The initial learning rate was set to ,batch size is 32, and training proceeded for 20 epochs.

Since the DUD - E and LIT - PCAB datasets are organized in different ways, the codes used during training also have some differences. Among them, train.py is the code for training the model on the DUD - E dataset, and train - LIT.py is the code for training the model on the LIT - PCBA dataset. A GPU can be used during training, and a CPU can be used as a substitute on devices without a GPU.
