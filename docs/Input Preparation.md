# Input Preparation

## How are MSA and Templates obtained?

### Why do we need MSA?

- Different versions of a certain type of protein existing across different species may have similar sequences and structures. By collecting these proteins together, we can observe how certain positions of a protein change with evolution.

- Each row of the MSA represents similar proteins from different species. Conservative patterns in columns can reflect the importance of requiring specific amino acids at that position. The relationships between different columns reflect the interactions between amino acids (if two amino acids physically interact, their amino acid changes during evolution may also be correlated). Therefore, MSA is often used to enrich the representation of a single protein.

### Why do we need Templates?

- Similarly, if the aforementioned MSA contains known structures, then it is very likely to help predict the structure of the current protein. Templates only focus on single-chain structures.

### How to obtain MSA?

- Use genetic search to find similar protein or RNA chains. One chain typically searches for N_msa (<16384) similar chains:
  
  ![image.png](../images/image.png)

- If there are multiple chains, and if they can be paired from the same species, then the MSA matrix may take this form:

  ![image.png](../images/image%201.png)

- Otherwise, it forms a diagonal matrix like this:

  ![image.png](../images/image%202.png)

### How to obtain Templates?

- Use Template Search. For the generated MSA, use HMM to search for similar protein sequences in the PDB database, then select the 4 highest quality structures from these matched sequences as templates.

### How to characterize Templates?

- Calculate the Euclidean distance between each token and token, using discretized distance representation (specifically, values are divided into 38 intervals, ranging from 3.15Å to 50.75Å, plus an additional interval representing distances exceeding 50.75Å).

- If certain tokens contain multiple atoms, select the central atom for distance calculation, for example, Cα is the central atom of amino acids, and C1 is the central atom of nucleotides.

- Templates only contain distance information on the same chain, ignoring interactions between chains.

## How to construct Atom-level representations?

- Constructing p and q: [Corresponding to Algorithm 5 AtomAttentionEncoder]
  - To construct atom-level single representation, we first need all atomic-level features. The first step is to construct a reference conformer for each amino acid, nucleotide, and ligand. The reference conformer can be looked up or calculated through specific methods, serving as local prior three-dimensional structural information.
  - c is the result after concatenating various features in the reference conformer and then performing a linear transformation. The shape of c is [C_atom=128, N_atoms], where C_atom is the feature dimension after linear transformation of each atom, and N_atoms is the total number of atoms in the sequence. c_l represents the features of the atom at position l, and c_m represents the features of the atom at position m.

    ![image.png](../images/image%203.png)
  - The atom-level single representation q is initialized through c:

    ![image.png](../images/image%204.png)
  - Then use c to initialize the atom-level pair representation p (atom-level pair representation). p represents the relative distances between atoms. The specific process is as follows:

    1. Calculate the distance between atomic reference three-dimensional coordinates, resulting in a three-dimensional vector.

       ![image.png](../images/image%205.png)
    2. Since the atomic reference three-dimensional coordinates are only obtained for their own reference conformations and are local information, distances are only calculated within tokens (amino acids, nucleotides, ligands, etc.). Therefore, we need to calculate a mask v. Only when chain_id and residue_idx are the same will coordinate position differences be calculated between each other. In this case, v is 1, otherwise v is 0.

       ![image.png](../images/image%206.png)
    3. Calculate p, with shape [N_atoms, N_atoms, C_atompair=16]:

       ![image.png](../images/image%207.png)

       - The p(l,m) vector has dimension [C_atompair]. The calculation method is to pass the d(l,m) three-dimensional vector through a linear layer to get a vector with dimension C_atompair, while multiplying by a scalar v(l,m) used for masking.
       - The inverse square of the distances 1/(1+||d(l,m)||^2) is a scalar, then undergoes linear transformation to become a [C_atompair] vector, still multiplied by the masking scalar v(l,m). p(l,m) = p(l,m) + this new vector.
       - Finally, p(l,m) is added with the mask scalar.

       ![image.png](../images/image%208.png)

       - p(l,m) also needs to add information from the original c, including information from c(:, l) and c(:, m). Both pieces of information first go through relu and then linear transformation, transformed into C_atompair vectors, and added to p(l,m).
       - Finally p(l,m) = p(l,m) + three-layer MLP(p(l,m))
- Updating q (Atom Transformer):
  1. Adaptive LayerNorm: [Corresponding to Algorithm 26 AdaLN]

     - The inputs are c and q, both with shape [C_atom, N_atoms]. c serves as a secondary input, mainly used to calculate the gamma and beta values for q, thus dynamically adjusting the layerNorm result of q through c.
     - Specifically, normal LayerNorm works like this:

       ![image.png](../images/image%209.png)
     - Then Adaptive Layer Norm works like this:

       ![image.png](../images/image%2010.png)

       - The formula is as follows:

         ![image.png](../images/image%2011.png)

         - Here a is q, and s is c.
         - During calculation, sigmoid(Linear(s)) is equivalent to the new gamma, and LinearNoBias(s) is equivalent to the new beta.
  2. Attention with Pair Bias [Corresponding to Algorithm 24 AttentionPairBias]:

     - The inputs are q and p, where q has shape [C_atom, N_atoms] and p has shape [N_atoms, N_atoms, C_atompair].
     - As a typical Attention structure, (Q, K, V) all come from q, with shape [C_atom, N_atoms].
       - Assuming N_head-headed attention, where a_i represents the vector q_i of the i-th atom in q, then for the h-th head and the i-th q vector, we get (q_h_i, k_h_i, v_h_i):

         ![image.png](../images/image%2012.png)

         - The dimension c is obtained as follows:

           ![image.png](../images/image%2013.png)
       - Pair-biasing: Where does it come from? Extract the i-th row from p, i.e., the relationship between the i-th atom and other atoms. Then p_i_j is the relationship between the i-th atom and the j-th atom, with vector shape [C_atompair]. In the formula, z_i_j is used to represent p_i_j.

         - z_i_j first undergoes LayerNorm in the C_atompair dimension, then undergoes linear transformation, reducing from C_atompair dimension to 1 dimension:

           ![image.png](../images/image%2014.png)
         - Then introduce this pair Bias before calculating softmax. At this time, for the vectors q_h_i and k_h_i of the i-th atom and j-th atom, first perform vector dot product and scale, then add a scalar b_h_i_j before softmax to get the weight A_h_i_j:

           ![image.png](../images/image%2015.png)
         - Then directly calculate the attention result for the i-th atom:

           ![image.png](../images/image%2016.png)
       - Gating: Obtain the vector of the i-th atom from q:

         - First perform a linear transformation, changing from c_atom dimension to Head dimension c, then directly calculate sigmoid, mapping it to a number between 0 and 1 as gating.

           ![image.png](../images/image%2017.png)
         - Then element-wise multiply with the attention result, finally concatenate all Heads, and finally undergo a linear transformation to get the attention result q_i with shape [C_atom].

           ![image.png](../images/image%2018.png)
       - Sparse attention: Since the number of atoms is much larger than the number of tokens, when calculating atom attention, from a computational perspective, we don't calculate attention of one atom to all atoms, but calculate local attention, called sequence-local atom attention. The specific method is:

         - When calculating attention, make the softmax result of positions (i,j) that we don't care about approximately 0, which is equivalent to setting the value at specific positions (i,j) to negative infinity before softmax, distinguished by the introduced beta_i_j:

           ![image.png](../images/image%2019.png)

           - If the distance between i and j satisfies the condition, then atom attention needs to be calculated, and beta_i_j is 0.
           - If the distance between i and j doesn't satisfy the condition, then attention doesn't need to be calculated, and beta_i_j is -10^10.
  3. Conditioned Gating: [Corresponding to Algorithm 24 AttentionPairBias]

     - The inputs are c and q, with shapes [C_atom, N_atoms] and [C_atom, N_atoms] respectively, and the output is q with shape [C_atom, N_atoms].
     - Here s_i in the formula is actually c_i. First perform a linear transformation, then calculate sigmoid, mapping each element in c_i to between 0 and 1, finally element-wise multiply with a_i, which is actually q_i, to get the updated q_i:

       ![image.png](../images/image%2020.png)
  4. Conditioned Transition: [Corresponding to Algorithm 25 ConditionalTransitionBlock]

     - The inputs are c, q, and p, with shapes [C_atom, N_atoms], [C_atom, N_atoms], and [N_atoms, N_atoms, C_atompair] respectively, and the output is q with shape [C_atom, N_atoms].
     - The last module of Atom Transformer, this is equivalent to the MLP layer in Transformer. It's called Conditional because it's contained between the Adaptive LayerNorm layer (step 3 above) and Conditional Gating (step 3 above). The only difference is whether the middle part is MLP or Attention.
       - The first step is still an Adaptive LayerNorm:

         ![image.png](../images/image%2021.png)
       - The second step is a swish:

         ![image.png](../images/image%2022.png)
       - The third step is Conditional Gating:

         ![image.png](../images/image%2023.png)

## How to construct Token-level representations?

- Constructing Token-level Single Sequence Representation
  - The input is q, atom-level single representation, with shape [C_atom, N_atoms]. The outputs are S_inputs and S_init, with shapes [C_token+65, N_tokens] and [C_token, N_tokens] respectively.
  - First, transform each atom's representation dimension C_atom through linear transformation to C_token, then apply relu activation function, then average all atom representations within the same token to get N_token vectors of C_token dimension. [From Algorithm 5 AtomAttentionEncoder]

    ![image.png](../images/image%2024.png)
  - Then for tokens with MSA features, concatenate residue_type (32) and MSA features (MSA properties: 32+1) to get S_input. [From Algorithm 2 InputFeatureEmbedder]

    ![image.png](../images/image%2025.png)
  - Finally, perform another linear transformation to convert S_input to S_init. [From Algorithm 1 MainInferenceLoop]

    ![image.png](../images/image%2026.png)

    - Note: Here C_s is C_token = 384
- Constructing Token-level Pair Representation [From Algorithm 1 MainInferenceLoop]
  - The input is S_init with shape [C_token=384, N_tokens], and the output is Z_init with shape [N_tokens, N_tokens, C_z=128].
  - To calculate Z_init_i_j, we need to get features of two specific tokens, perform linear transformations on them separately, then add them to get the first z_i_j, with vector length transformed from C_tokens (i.e., C_s=384) to C_z=128.

    ![image.png](../images/image%2027.png)
  - Then add relative position encoding at position (i,j) of z:

    ![image.png](../images/image%2028.png)

    - Detailed explanation of RelativePositionEncoding: Note that both i and j here refer to tokens. [From Algorithm 3 RelativePositionEncoding]

      ![image.png](../images/image%2029.png)

      - a_residue_i_j: Residue relative position information:
        - If tokens i and j are in the same chain, then d_residue_i_j is the relative position difference between residue i and residue j, ranging from [0, 65].
        - If tokens i and j are not in the same chain, then d_residue_i_j = 2*r_max+1 = 65
        - a_residue_i_j is a one-hot encoding of length 66, where the position of 1 is the value of d_residue_i_j.
      - a_token_i_j: Token relative position information:
        - If tokens i and j are in the same residue (for modified amino acids or nucleotides, one atom is one token), then d_token_i_j is the difference between different residue indices, ranging from [0, 65].
        - If tokens i and j are not in the same residue, take the maximum value d_token_i_j = 2*r_max+1 = 65
        - a_token_i_j is a one-hot encoding of length 66, where the position of 1 is the value of d_token_i_j.
      - a_chain_i_j: Chain relative position information:
        - If tokens i and j are not in the same chain, then d_chain_i_j is the index difference between chains, ranging from [0, 5].
        - If tokens i and j are in the same chain, then d_chain_i_j is set to the maximum value 5.
        - a_chain_i_j is a one-hot encoding of length 6, where the position of 1 is the value of d_chain_i_j.
      - b_same_entity_i_j: If tokens i and j are in the same entity (completely identical amino acid sequences are unique entities with unique entity ids), it's 1, otherwise 0.
      - Finally, concatenate [a_residue_i_j, a_token_i_j, b_same_entity_i_j, a_chain_i_j] to get a vector of length 66+66+1+6 = 139 = C_rpe.
      - Then undergo another linear transformation to transform the vector dimension to C_z=128 dimension.
  - Finally, add token bond information, which after linear transformation is added to z_i_j to get the final result.

    ![image.png](../images/image%2030.png)