# Loss Function

The final LossFunction formula is as follows:

$\mathcal{L}\_{\text{loss}} = \alpha\_{\text{confidence}} \cdot L\_{\text{confidence}} + \alpha\_{\text{diffusion}} \cdot \mathcal{L}\_{\text{diffusion}} + \alpha\_{\text{distogram}} \cdot \mathcal{L}\_{\text{distogram}}$

Where, $L\_{confidence}= \mathcal{L}\_{\text{plddt}} + \mathcal{L}\_{\text{pde}} + \mathcal{L}\_{\text{resolved}} + \alpha\_{\text{pae}} \cdot \mathcal{L}\_{\text{pae}}$

$\mathcal{L}\_{\text{loss}} = \alpha\_{\text{confidence}} \cdot \left( \mathcal{L}\_{\text{plddt}} + \mathcal{L}\_{\text{pde}} + \mathcal{L}\_{\text{resolved}} + \alpha\_{\text{pae}} \cdot \mathcal{L}\_{\text{pae}} \right) + \alpha\_{\text{diffusion}} \cdot \mathcal{L}\_{\text{diffusion}} + \alpha\_{\text{distogram}} \cdot \mathcal{L}\_{\text{distogram}}$

- L_distogram: Used to evaluate whether the predicted token-level distogram (i.e., distances between tokens) is accurate.
- L_diffusion: Used to evaluate whether the predicted atom-level distogram (i.e., relationships between atoms) is accurate, and also includes some additional terms, including prioritizing relationships between nearby atoms and processing atoms in protein-ligand bonds.
- L_confidence: Used to evaluate the accuracy of the model's self-awareness about which parts of its predicted structure are accurate or inaccurate.

## $L_{distogram}$

- Although the output result is atom-level three-dimensional coordinates, the L_distogram loss here is a token-level metric, representing the accuracy of the model's prediction of distances between tokens. But since atomic three-dimensional coordinates are obtained, to calculate token three-dimensional coordinates, the three-dimensional coordinates of the central atom of this token are directly used as the token's three-dimensional coordinates.
- Why evaluate whether the distance prediction between tokens (or atoms) is accurate, rather than directly evaluating whether the coordinate prediction of tokens (or atoms) is accurate? → The most fundamental reason is that the coordinates of tokens (or atoms) in space may change with the rotation or translation of the entire structure, but the relative distances between them will not change. Such a loss can reflect the rotational and translational invariance of the structure.
- What is the specific formula?

  $\mathcal{L}\_{\text{dist}} = -\frac{1}{N\_{\text{res}}^2} \sum_{i,j} \sum_{b=1}^{64} y\_{ij}^b \log p\_{ij}^b$

  - Here y_b_i_j refers to uniformly dividing the distance between the i-th token and j-th token into 64 buckets (from 2Å to 22Å). y_b_i_j refers to one of these 64 buckets. Here y_b_i_j uses one-hot encoding to represent that the actual result falls into a specific bucket.
  - p_b_i_j refers to the probability that the distance value between the i-th token and j-th token falls into a certain bucket, which is a result after softmax.
  - For any token pair (i,j), find the difference between its predicted distance and actual distance using cross-entropy:
    - Calculate $\sum_{b=1}^{64} y_{ij}^b \log p_{ij}^b=\log p_{ij}^{\text{target-bin}}$
  - For all token pairs, calculate the average loss:
    - Calculate $-\frac{1}{N\_{\text{res}}^2} \sum_{i,j} \log p\_{ij}^{\text{target-bin}}$ to get the final L_distogram loss value.

## $L_{diffusion}$

- Diffusion training process: (The red box part in the figure represents the Diffusion training settings. This figure shows the overall training settings of AlphaFold3. Note that the distogram loss part is ignored)

  ![image.png](../images/image%2071.png)

  - In the diffusion training process, it first uses the output results of the trunk as input, including original atomic features f*, updated token-level pair representations, token-level single representations, etc.
  - During training, the Diffusion part will use a larger batch_size compared to the trunk part. After each sample passes through the trunk module, it will generate 48 related but different three-dimensional structures as input to the diffusion module. These structures are all generated based on real structures (real three-dimensional structures from training samples), but will undergo random rotation and translation, and add different degrees of noise. This approach is to obtain a large number of (noisy structure, target structure) pairs, allowing the diffusion model to learn how to denoise.
  - Since the noise size is random, it's equivalent to generating a noisy structure from time step t (belonging to [0,T], based on different noise sizes, t may be close to T-1 time step or close to 0 time step), then hoping that the model approaches T time step after one diffusion module update, i.e., a completely noise-free state.
  - Finally, for these 48 different results, compare them with Ground Truth results, calculate loss (L_diffusion), and backpropagate to optimize the model parameters (is this only optimizing the diffusion module or also optimizing the trunk module?).
- Diffusion loss function construction: Note that loss function calculation is actually performed at the atomic level here.

  - $L_{MSE}$: Used to calculate the weighted Mean Squared Error of the difference between target atomic coordinates and predicted atomic coordinates. Here we know the target atomic sequence for three-dimensional coordinates: {x_GT_l}, and the predicted three-dimensional coordinate atomic sequence {x_l}. Calculate the difference between these two three-dimensional structures. The specific calculation method is as follows:
    - First perform rigid alignment on the target three-dimensional structure, which is equivalent to aligning the overall position and orientation of the two structures, allowing comparison under the same reference coordinate system. This way, the comparison error is the structural difference itself, not caused by rotation or position offset. This also explains why when calculating L_distogram earlier, without rigid alignment, we directly compared the difference between token distances in true and predicted values, rather than directly comparing the difference in token coordinates.

      ![image.png](../images/image%2072.png)
    - Then calculate the L_mse value: $L_{MSE} = \frac{1}{3} \cdot \text{mean}_l \big( w_l \| \tilde{x}_l - x_l^{GT-aligned} \|^2 \big)$
    - Note this is a weighted Mean Squared Error: $w_l = 1 + f_l^{\text{is-dna}} \alpha^{\text{dna}} + f_l^{\text{is-rna}} \alpha^{\text{rna}} + f_l^{\text{is-ligand}} \alpha^{\text{ligand}}$, where $\alpha^{\text{dna}} = \alpha^{\text{rna}} = 5, \alpha^{\text{ligand}} = 10$. Here, higher weights are set for RNA/DNA and ligand, meaning higher accuracy requirements for predictions of these atoms.
  - $L_{bond}$: Loss function to ensure reasonable bond lengths between ligands and main chains.
    - Why is this loss needed? The reason is that diffusion models can recover a model with correct overall structure but insufficient detail precision, such as certain chemical bonds becoming too long or too short. At the same time, ligands are like small ornaments hanging on the protein chain - you don't want these ornaments to be too long or too short, while peptide bonds between protein amino acids have basically stable lengths, and the internal atomic arrangement of the main chain itself has relatively strong constraints.
    - So the calculation method here is: $\mathcal{L}\_{\text{bond}} = \text{mean}\_{(l,m) \in \mathcal{B}} \left( \left\| \vec{x}\_l - \vec{x}\_m \right\| - \left\| \vec{x}\_l^{\text{GT}} - \vec{x}\_m^{\text{GT}} \right\| \right)^2$, where $\mathcal{B}$ refers to a series of atom pairs (l is the starting atom index, m is the ending atom index), representing protein-ligand bonds. This is equivalent to calculating the average difference between target bond length and actual bond length.
    - Essentially, this is also an MSE loss.
  - $L_{smooth-LDDT}$: Loss for comparing the difference between predicted atomic pair distances and actual atomic pair distances (Local Distance Difference Test), with emphasis on the accuracy of distance predictions between nearby atoms.
    - The specific calculation pseudocode is:

      ![image.png](../images/image%2073.png)

      - The first two steps calculate distances between any two atoms, including predicted and actual values.
      - Next, calculate the absolute difference $\delta_{lm}$ between predicted and actual distances for atom pair (l,m).
      - Then calculate a score distributed between [0,1], which is used to measure whether $\delta_{lm}$ can pass the (Local Distance Difference Test).

        - Four Tests are set here, each using different thresholds. If $\delta_{lm}$ is within the set threshold range, then the distance prediction for atom pair (l,m) is considered to pass the Test, and the Test score will be greater than 0.5, otherwise it fails and is less than 0.5.
        - So each Test sets different thresholds (4, 2, 1, and 0.5 Å respectively), using sigmoid function to implement: sigmoid(threshold - $\delta_{lm}$). The function curves for these four Tests are shown below:

          ![Individual Sigmoid Terms in Smooth LDDT.png](../images/Individual_Sigmoid_Terms_in_Smooth_LDDT.png)
        - Then average the results of these four Tests to get score $\epsilon_{lm}$. This is the curve of this score - you'll find that the closer to 0, the closer this score is to 1, otherwise it approaches 0.

          ![Smooth LDDT Component vs Distance Difference.png](../images/Smooth_LDDT_Component_vs_Distance_Difference.png)
      - Then, to make this calculation score mainly examine distances between nearby atoms, atom pairs with very far actual distances are not included in the loss calculation (c_l_m=0). That is, nucleotide atom pairs with actual distances greater than 30Å and non-nucleotide atom pairs with actual distances greater than 15Å are not included.
      - Finally, calculate the mean of $\epsilon_{lm}$ scores for atom pairs where c_l_m is not 0 as the lddt value. The closer this value is to 1, the more accurate the average atom pair prediction. Convert it to loss as 1-lddt.
  - Finally, $\mathcal{L}\_{\text{diffusion}} = \frac{\hat{t}^2 + \sigma\_{\text{data}}^2}{(\hat{t} + \sigma\_{\text{data}})^2} \cdot \left( \mathcal{L}\_{\text{MSE}} + \alpha\_{\text{bond}} \cdot \mathcal{L}\_{\text{bond}} \right) + \mathcal{L}\_{\text{smooth-lddt}}$
    - Here $\sigma_{data}$ is a constant determined by data variance, set to 16 here.
    - Here t^ is the sampled noise level during training, specifically calculated as $\hat{t}=\sigma_{\text{data}} \cdot \exp\left( -1.2 + 1.5 \cdot \mathcal{N}(0, 1) \right)$
    - Here $\alpha_{bond}$ is 0 during initial training and 1 during fine-tuning.

## $L_{confidence}$

- The role of this last loss is not to improve the accuracy of the model's structure prediction, but to help the model learn how to evaluate the accuracy of its own predictions. This loss is also a weighted sum of four different losses used to evaluate self-accuracy.
- The specific formula is: $L\_{confidence}= \mathcal{L}\_{\text{plddt}} + \mathcal{L}\_{\text{pde}} + \mathcal{L}\_{\text{resolved}} + \alpha\_{\text{pae}} \cdot \mathcal{L}\_{\text{pae}}$
- Mini-Rollout explanation:

  ![image.png](../images/image%2074.png)

  - **Principle**: Normally, to calculate the model's confidence in the generated three-dimensional structure, you need to obtain the model's final generated three-dimensional structure for calculation, similar to AF2's approach. But for AF3, the diffusion module cannot directly generate the final denoised result in a single iteration. Therefore, a mini-rollout mechanism is introduced here: during training, perform a fixed number of iterations (20 times) on the Diffusion module, allowing the model to quickly generate an approximate protein structure prediction from random noise. Then use this temporary prediction to calculate evaluation metrics and train the confidence head.
  - **Gradient blocking**: Note that mini-rollout here does not backpropagate gradients (as shown by the red STOP sign in the figure. Not used to optimize the Diffusion module nor the Network trunk module), because the main goal of calculating L_confidence is to optimize the model's ability to evaluate the quality of generated structures, i.e., optimize the performance of the confidence module itself. This design ensures that the training objectives of the diffusion module (single-step denoising) and confidence head (structure quality measurement) are independent of each other, avoiding conflicts caused by inconsistent training objectives. It also ensures that the training objectives of the Trunk module (providing better feature representations, providing rich and general feature representations for subsequent structure generation) and confidence head (structure quality measurement) can be independent of each other.
- Note that these confidence losses are only used for PDB datasets (not applicable to any distillation datasets, as distillation dataset structures are predicted structures, not real structures); filtering is also performed on datasets, selecting only filtered resolution (between 0.1Å and 4Å) real structures for confidence loss training, to ensure the model can learn error distributions close to real physical structures.
- The meaning of each loss is explained in detail below:

  - Predicted Local Distance Difference Test (pLDDT): Average confidence of each atom. (Note that AlphaFold2 is average confidence of each residue)

    - Calculate LDDT for a single atom: $lddt_l$ (during training):

      - The goal here is to estimate the difference between predicted and real structures, specifically for a particular atom.
      - So the calculation formula is set as follows:

        ![image.png](../images/image%2075.png)

        - Where $d\_{lm}$ is the distance between atoms l and m predicted by mini-rollout.
        - ${m}\in{R}$, the selection of atom m is based on the real three-dimensional structure of this training sequence: 1) m's distance is within a certain nearby range of l (30Å or 15Å, depending on m's atom type); 2) m only selects atoms located on polymers (small molecules and ligands are not considered); 3) only one atom per token is considered. For atoms in standard amino acids or nucleotides, m uses their representative atoms ($C_\alpha$ or $C_1$) for representation.
        - Then for each pair (l,m), perform LDDT (Local Distance Difference Test): $\frac{1}{4} \sum_{c \in \{0.5, 1, 2, 4\}} d\_{lm} < c$. If l and m are relatively close in real distance, they should also be close enough in prediction results. Four thresholds are set here - if all are satisfied, LDDT is 1; if none are satisfied, it's 0.
        - Finally, sum all LDDT values calculated for m near l to get an $lddt_l$ value for atom l, whose magnitude can measure the difference between the model's predicted structure and real structure at atom l. Note this is an unnormalized value.
    - Calculate confidence head output probability distribution of this atom's LDDT: $p_l^{\text{plddt}}$ (during training and prediction):

      - The specific confidence calculation process is temporarily ignored here (will be detailed later). What needs to be known is that $p_l^{\text{plddt}}$ here is calculated through the confidence head at atom l, providing an estimate of the distribution of $lddt_l$ values.
      - Here $p_l^{\text{plddt}}$ is a 50-dimensional vector, dividing 0~100 into 50 bins, which is a softmax result predicting the probability distribution of $lddt_l$ values falling into specific bins.
      - Note that the calculation here involves no real structure at all, and is entirely based on predictions from previous trunk-related representations.
    - Calculate the entire $L_{plddt}$ (during training):

      - So the optimization goal of this loss is not to maximize $lddt_l$, but to more accurately predict $lddt_l$.
      - Rather, the actual $lddt_l$ value and the model's predicted $lddt_l$ distribution should always be aligned: if the actual $lddt_l$ value is low (model structure prediction is inaccurate), then in the model's predicted $lddt_l$ distribution $p_l^{\text{plddt}}$ result, the probability of falling into smaller value bins should be higher; if the actual $lddt_l$ value is high (model structure prediction is accurate), then in the model's predicted $lddt_l$ distribution $p_l^{\text{plddt}}$ result, the probability of falling into larger value bins should be higher.
      - So use cross-entropy loss to align the difference between these two, which can ensure that the model's real LDDT distribution and predicted LDDT distribution are as consistent as possible: $\sum_{b=1}^{50} \text{lddt}_l^b \log p_l^b$.
      - Finally, because we need to calculate the overall loss, average over all atoms to get the final calculation method:

        ![image.png](../images/image%2076.png)
    - Calculate pLDDT value (during prediction):

      - Additionally, when predicting, when the model outputs the pLDDT value of a single atom, the calculation method is: $p_l^{\text{plddt}} * V_{bin}$, getting a scalar between 0~100, representing the model's predicted value for the lddt of atom l at the current position. When atoms around this atom are all relatively close to it, the lddt value is large, representing higher confidence in the model's prediction of the current atom l position, otherwise lower confidence.
      - The reason is that after optimization by the previous loss function, $p_l^{\text{plddt}}$ is a distribution with good evaluation ability for the prediction effect of atom l. So we can trust $p_l^{\text{plddt}}$'s estimation of lddt distribution, which can be equivalent to calculating the expected value to get the lddt prediction value.
  - Predicted Aligned Error (PAE): Confidence prediction of alignment error between token pairs (calculated by atomic pair distances).

    - Some concept and method explanations:

      - **reference frame:** A token's reference frame is represented using coordinates of three atoms: $\Phi_i = (\vec{a}_i, \vec{b}_i, \vec{c}_i)$. The role of this frame is to define a local reference coordinate system for token i, used to establish connections with token j. For different tokens, the selection of three atoms for the reference frame is different:

        - For protein tokens or residues, their reference frame is: $(\text{N}, \text{C}^\alpha, \text{C})$
        - For DNA or RNA tokens, their reference frame is: $(\text{C1}', \text{C3}', \text{C4}')$
        - For other small molecules, their token may contain only one atom, so b_i is chosen as this atom itself, then the nearest atom is chosen as a_i, and the second nearest atom as c_i.
        - Exception: If the chosen three atoms are almost in a straight line (the angle between them is less than 25 degrees), or these three atoms cannot be found in the actual chain (e.g., sodium ion has only one atom), then this frame is defined as an invalid frame and does not participate in subsequent PAE calculations.
      - $\text{expressCoordinatesInFrame}(\vec{x}, \Phi)$: Express coordinates of atom $\vec{x}$ in the $\Phi$ coordinate system.

        ![image.png](../images/image%2077.png)

        - Rough explanation of this algorithm:
          - First, get coordinates a, b, c of three reference atoms from $\Phi$. Treat b as the origin of the new coordinate system.
          - Then, from directions b to a and b to c, construct an orthonormal basis (e_1, e_2, e_3).
          - Finally, project x onto this new basis to get x_transformed, which is the coordinate of x in the new coordinate system $\Phi$.
        - Detailed explanation of this algorithm:
          - Given coordinates of three reference atoms, we need to construct an orthogonal three-dimensional coordinate system with atom b's coordinates as the origin.
          - Calculate w1 and w2, which are **unit vectors** in the direction from b to a and from b to c respectively.
          - Then calculate the orthogonal basis:
            - e1 can be seen as a direction that "bisects the angle" between a and c.
            - e2 is a direction after subtracting w1 and w2. Since w1 and w2 are both unit vectors, this vector is orthogonal to e1 and also in the same plane.
            - e3 is obtained by cross product of e2 and e1, getting the third basis vector perpendicular to both, thus forming three complete orthogonal bases.
            - After completing this step, e1, e2, e3 form a right-handed orthonormal basis with b as origin.
          - Finally project x onto this coordinate system:
            - First, translate x so that b becomes the origin.
            - Then, perform projection, calculating d's projection on each basis vector, i.e., (d*e1, d*e2, d*e3).
            - Finally, get the new coordinates of x in coordinate system $\Phi$: x_transformed.
      - $\text{computeAlignmentError}(\{\vec{x}\_i\}, \{\vec{x}\_i^\text{true}\}, \{\Phi_i\}, \{\Phi_i^\text{true}\}, \epsilon = 1e^{-8} \, \text{Å}^2)$: Calculate alignment error between token i and token j.

        ![image.png](../images/image%2078.png)

        - Input:
          - x_i refers to predicted coordinates of the representative atom for token i, x_true_i refers to real coordinates of the representative atom for token i.
          - $\Phi_i$ refers to predicted reference frame for token i, $\Phi_i^\text{true}$ refers to real reference frame for token i.
        - Calculation process:
          - Predicted result of relationship between token pair (i, j): In token i's reference frame local coordinate system, calculate coordinates of token j's representative atom in this coordinate system, equivalent to calculating the relative relationship of token j with respect to token i.
          - Real result of relationship between token pair (i, j): In token i's reference frame local coordinate system, calculate coordinates of token j's representative atom in this coordinate system, equivalent to calculating the relative relationship of token j with respect to token i.
          - Calculate alignment error, i.e., the difference between predicted relative position and real relative position, using Euclidean distance for calculation. If e_i_j is relatively small, then the predicted relationship between tokens i and j aligns well with the real relationship between tokens i and j, otherwise the alignment is poor.
          - Note that (i,j) here is not commutative, e_i_j and e_j_i are different.
    - PAE Loss calculation process:

      - $\mathbf{p}_{ij}^{\text{pae}}$ calculated through confidence head is a b_pae=64 dimensional vector, representing the probability that e_i_j falls into 64 bins (from 0Å to 32Å, with 0.5Å steps).
      - To make the distribution of $\mathbf{p}\_{ij}^{\text{pae}}$ closer to the actual value of e_i_j, use cross-entropy loss function to align the two, so that $\mathbf{p}\_{ij}^{\text{pae}}$ can better predict the actual value of e_i_j. (Note: The loss design here is not to minimize the value of e_i_j, which might be for better structural prediction accuracy; but to better align the predicted probability $\mathbf{p}\_{ij}^{\text{pae}}$ with the result of e_i_j through cross-entropy loss, thus better predicting the magnitude of e_i_j; larger e_i_j indicates the model believes there is greater uncertainty in the relative conformation of these two positions, smaller e_i_j means more confidence in the relative conformation of those two positions)
      - So the final PAE loss is defined as: (Note that e_b_i_j here is different from the previous e_i_j. If e_i_j falls in the corresponding bin b, then the corresponding e_b_i_j is 1, otherwise e_b_i_j is 0)

        ![image.png](../images/image%2079.png)
    - If calculating PAE_i_j value in prediction, calculate through expectation method.

      - Take the center values of 64 discrete bins, then multiply by the predicted probability p_b_i_j at each position (i.e., the probability that e_i_j value falls in this bin), to get an expected value for e_i_j:

        ![image.png](../images/image%2080.png)
  - Predicted Distance Error (PDE): Confidence prediction of absolute distances between representative atoms of token pairs.

    - Besides alignment error, the model also needs to predict prediction errors of absolute distances between important atoms.
    - The calculation method for distance error here is relatively simple, as follows:

      - First, calculate the absolute distance between representative atoms of token i and token j predicted by the model: $d\_{ij}^{\text{pred}}$
      - Then, calculate the real absolute distance between representative atoms of token i and token j: $d\_{ij}^{\text{gt}}$
      - Finally, directly calculate the absolute difference between the two: $e\_{ij} = \left| d\_{ij}^{\text{pred}} - d\_{ij}^{\text{gt}} \right|$
    - Similarly, the result of $\mathbf{p}_{ij}^{\text{pae}}$ predicted through confidence head is also a 64-dimensional vector, representing the probability that e_i_j falls into 64 bins (from 0Å to 32Å, with 0.5Å steps).
    - Similarly, then align the two through cross-entropy loss to get L_pde:

      ![image.png](../images/image%2081.png)
    - Similarly, in prediction, use expectation method to calculate a token-pair's pde value: ($\Delta_b$ is the interval center value)

      ![image.png](../images/image%2082.png)
  - Experimentally Resolved Prediction: Predict whether an atom can be experimentally observed

    - This is a prediction confidence value with atom index l, used to indicate whether the current atom l can be correctly experimentally observed.
    - y_l refers to whether the current atom is correctly experimentally resolved, which is a 2-dimensional 0/1 value; p_l is a 2-dimensional vector from the confidence head, which is a softmax result, representing the model's prediction of whether the current atom l is correctly resolved.
    - The final optimization goal is to predict whether the current atom can be correctly experimentally resolved, so the loss function is:

      ![image.png](../images/image%2083.png)
- Confidence Head calculation: The goal of Confidence Head is to further generate a series of confidence distributions (pLDDT, PAE, PDE, resolved, etc.) based on previous model representations and predictions, which can be used for subsequent confidence loss calculations (or directly for model output predictions)

  - Confidence Head inputs:

    - Token-level single embedding features {s_inputs_i} from the initial InputFeatureEmbedder.
    - Token-level single embedding {s_i} and token-level pair embedding {z_i_j} from the backbone network.
    - Mini-rollout predicted structure from diffusion module: {x_pred_l}.
  - Algorithm calculation process analysis:

    ![image.png](../images/image%2084.png)

    1. Update token-level pair embedding z_i_j, adding information projected from initial single embedding.
    2. Calculate the distance between three-dimensional coordinates of representative atoms (atom index l_rep(i)) of tokens i and j predicted by the model, denoted as d_i_j.
    3. Discretize the value of d_i_j to intervals defined by v_bins, calculate its one-hot representation, then through linear transformation, update to token-level pair embedding.
    4. Continue to update token-level single representation {s_i} and pair representation {z_i_j} through Pairformer, equivalent to letting these two types of representations interact and strengthen each other for several rounds, obtaining final {s_i} and {z_i_j}.
    5. Calculate PAE confidence probability, with final result being a b_pae=64 dimensional vector. Since PAE is actually token-token representation (although actually calculating distance between representative atoms and frames), use {z_i_j} for linear transformation then directly calculate softmax to obtain this confidence probability, representing the probability that its PAE value falls in each of 64 intervals. (Note: i and j are not commutative here)
    6. Calculate PDE confidence probability, with final result being a b_pde=64 dimensional vector. Similarly, PDE is also token-token representation (actually calculating absolute distance between representative atoms), use information from z_i_j and z_j_i for fusion, then linear transformation and direct softmax to obtain confidence probability, representing the probability that PDE value falls in each of 64 intervals. (Note: i and j are commutative here)
    7. Calculate pLDDT confidence probability (Note: pLDDT confidence probability here is the value for each atom, indexed by atom index l rather than token index i.)
       1. s_i(l) here means: get the token-level single embedding corresponding to token i that atom l corresponds to.
       2. LinearNoBias_token_atom_idx(l)( … ), the function of this function is that for different atoms l, the matrix used for linear transformation is different. Get the corresponding weight matrix through token_atom_idx(l), with matrix shape [c_token, b_plddt], then right multiply with s_i(l) with shape [c_token] to get final vector [b_plddt].
       3. Finally perform softmax to get pLDDT confidence probability, with b_plddt=50, which is a 50-dimensional vector indicating the probability that lddt value falls within these 50 bin ranges.
    8. Calculate resolved confidence probability (Note: resolved confidence probability here is also the value for each atom, same as above): The result after calculation and softmax is a 2-dimensional vector, predicting the confidence of whether the current atom can be experimentally resolved.