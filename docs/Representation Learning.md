# Representation Learning

## Template Module

- What are the inputs to the Template Module?

  - token-level pair representation { z_i_j } and features containing template information {**f**}.
- How are Template features constructed? [From Algorithm 16 TemplateEmbedder]

  ![image.png](../images/image%2031.png)

  1. For template t, b_template_backbone_frame_mask_i_j is calculated through the template_backbone_frame_mask values at positions i and j of the t-th template. Only when both positions i and j contain all atoms that can be used to calculate the backbone, this mask result is 1, otherwise 0.
  2. Similarly, only when the template_pseudo_beta_mask at positions i and j of the t-th template are both 1, this b_template_pseudo_beta_mask_i_j mask is 1, otherwise 0. This means whether the central atom has coordinates at positions i and j.
  3. a_t_i_j calculates related features at positions i and j of the t-th template, concatenating the last dimension of template_distogram (representing distances between two tokens, with shape [N_templ, N_token, N_token, 39]) with b_template_backbone_frame_mask_i_j (1-dimensional scalar), the last dimension of template_unit_vector (representing unit vectors with direction between two tokens, with shape [N_templ, N_token, N_token, 3]), and b_template_pseudo_beta_mask_i_j (1-dimensional scalar) to form a 44-dimensional vector.
  4. If the token at position i and the token at position j are not on the same chain, then the a_t_i_j vector is set to an all-zero vector.
  5. a_t_i_j is then concatenated with the residue type (one-hot encoding) at position i and the residue type information at position j of template t, resulting in an a_t_i_j vector of [44+32+32 =108]. The shape of {a_t_i_j} is [N_templ, N_token, N_token, 108].
- Then perform the same operations for each template: [From Algorithm 16 TemplateEmbedder]

  ![image.png](../images/image%2032.png)

  1. First define u_i_j to store calculation results, with shape [N_token, N_token, c] for {u_i_j}.
  2. For each template, perform the same operations: perform linear transformation on a_t_i_j from 108 dimensions → c dimensions, perform LN on z_i_j first, then perform linear transformation from c_z dimensions → c dimensions, then for each template's a_t_i_j, add the same transformed z_i_j result to get v_i_j with c dimensions.
  3. Then pass v_i_j through PairformerStack for calculation, add the result to v_i_j as the v_i_j result, with c dimensions.
  4. Finally, after performing LN on v_i_j, add all templates to the u_i_j vector, with c dimensions.
- Finally, average the u_i_j results by dividing by the number of templates, then activate, and undergo a linear transformation to get the final result u_i_j, with shape [N_token, N_token, c] for {u_i_j}. [From Algorithm 16 TemplateEmbedder]

  ![image.png](../images/image%2033.png)

## MSA Module

![image.png](../images/image%2034.png)

- The goal of this module: Simultaneously update MSA and Pair Representation features and make them influence each other. First use Outer Product Mean to let MSA representation update Pair Representation, then use Pair Representation through row-wise gated self-attention using only pair bias to update MSA representation. Finally, Pair representation undergoes a series of triangular calculations and is ultimately updated.
- Inputs:

  - MSA representation (will be subsampled by rows, randomly selecting only a small portion of samples)
    - f_msa: Shape [N_msa, N_token, 32], original N_msa MSA samples, each position has 32 possibilities.
    - f_has_deletion: Shape [N_msa, N_token], original information indicating whether there is a deletion to the left of each position.
    - f_deletion_value: Shape [N_msa, N_token], original information indicating raw deletion counts, with values between [0,1].
  - S_inputs_i: Initial token-level single sequence features, shape [N_token, C_token+65], where 65 is 32+32+1, including the restype of the current token and some MSA features, including the distribution of various restype types at position i and the deletion mean value at position i. If the current token has no MSA information, all 65 dimensions are 0.
  - token-level Pair Representation: {z_ij}, shape [N_token, N_token, C_z=128]
- Output: Updated token-level Pair Representation: {z_ij}, shape [N_token, N_token, C_z=128]
- Look directly at the pseudocode:

  ![image.png](../images/image%2035.png)

  1. First, concatenate f_msa_Si, f_has_deletion_Si, and f_deletion_Si to get a 32+1+1=34 vector: **m_Si**. Note that S and i here are both subscripts, S represents the S-th row (S-th sample) in N_msa, and i represents the i-th position in each row.
  2. Then use the SampleRandomWithoutReplacement function to subsample N_msa, where {S} represents the set of all possible indices of N_msa, and {s} represents the set of possible indices after subsampling.
  3. Then **m_si** represents the s-th sample and i-th position for the subsampled samples, then perform linear transformation, changing dimension from 34→C_m=64.
  4. Then add the result of linear transformation of {S_inputs_i} to get a new MSA feature **m_si**. Here S_inputs_i also includes MSA-related features at each position (if any), with dimension C_token+65.
  5. Then loop for N_block blocks:
     1. Perform OuterProductMean calculation on MSA information m_si and integrate it into pair representation:

        1. Input {m_si} has shape [n_msa, N_tokens, C_m], the specific algorithm is as follows:

           ![image.png](../images/image%2036.png)

           1. First, perform LN on m_si, {m_si} has shape [n_msa, N_tokens, C_m]
           2. Then perform linear transformation to get a_si and b_si, linear transformation is from C_m=64 → c=32
           3. For each s, calculate the outer product of a_si and b_sj, which is equivalent to calculating the relationship between position i and position j for this MSA sample s through outer product.
              1. The outer product of a vector of length c with a vector of length c results in a [c, c] matrix,
              2. Then average s [c, c] matrices at each position (i,j),
              3. Finally, flatten this matrix into a one-dimensional vector to get o_ij with shape c*c.
           4. Finally, undergo another linear transformation, changing c*c → c_z=128 dimension, to get the final z_ij, i.e., information between positions calculated through MSA.

           Note 1: Here, through the OuterProductMean method, MSA representation is integrated into pair representation. For the same MSA sequence, the relationship between any two positions is obtained through outer product, then the results of these two positions for all MSA sequences are averaged to get evolutionary relationship information between any two positions and integrate it into pair representation.

           Note 2: Note that calculations are only performed within evolutionary sequences, and information between evolutionary sequences is integrated only once through averaging, avoiding complex calculations between evolutionary sequences in AF2.
     2. Use updated pair representation and m_si to update m_si (MSA features): MSA row-wise gated self-attention using only pair bias

        1. Input {m_si} has shape [n_msa, N_tokens, C_m=64], {z_ij} has shape [N_token, N_token, C_z=128]. Output is {m_si} with shape [n_msa, N_tokens, C_m=64]. The specific algorithm is as follows:

           ![image.png](../images/image%2037.png)

           1. First perform LN on MSA features m_si.
           2. Then perform multi-head linear transformation on m_si to get H_head heads v_h_si, linear transformation dimension from C_m→ c.
           3. First perform LN on pair representation features z_ij, then perform multi-head linear transformation to get b_h_ij, dimension from C_z→1.
           4. Perform multi-head linear transformation on m_si, dimension from C_m→c, then calculate its sigmoid value to get g_h_si for subsequent gating.
           5. Perform softmax on b_h_ij along the j direction to get weight w_h_ij with dimension 1.
           6. What's difficult to understand here is how to get o_h_si. Here v_h_sj and w_h_ij are element-wise multiplied in the j direction, then summed to get the intermediate result of o_h_si.
              1. That is, for the matrix {w_h_ij} with shape [N_token, N_token], take the elements of its i-th row [N_token].
              2. For the matrix {v_h_sj} with shape [n_msa, N_token, c], take the elements of its s-th row [N_token, c].
              3. Perform element-wise multiplication and sum them up to get a c-dimensional vector, whose position is the s-th row and i-th column of the original {m_si} matrix.
              4. o_h_si is then element-wise multiplied by g_h_si in the c dimension for gating.
           7. Finally, for o_h_si, concatenate its H_head heads to get a vector of length c*H_head, then undergo linear transformation to get the final result m^_si.
        2. Note that this part updates MSA representation through pair representation. The update method is that for each MSA sequence, its updates are independent of each other. Then use the relationships between positions in pair representation to construct weights, which is equivalent to performing self-attention on each position in m_si, introducing information from pair representation.
     3. {m_si} undergoes another transition layer and then serves as the input for {m_si} in the next block.
     4. pair representation {z_ij} undergoes a series of triangular calculations and transitions, then serves as the input for {m_si} in the next block.

## Pairformer Module

![image.png](../images/image%2038.png)

- First understand what the Pairformer module mainly does: Pair Representation will undergo triangular updates and triangular attention calculations, and is used to update Single Representation. Unlike AF2, here Single Representation does not affect Pair Representation.
- Input and output: The inputs are token-level pair representation {z_ij} and token-level single representation {s_i}, with shapes [N_token, N_token, C_z=128] and [N_token, C_s=C_token=384] respectively.
- Why focus on triangular relationships? (Why look at Triangles?)

  - The triangle inequality states: The sum of any two sides of a triangle is greater than the third side. In pair representation, the relationship between any two tokens is represented. To simplify understanding, we can view it as the distance between any two amino acids in a sequence. Then z_ij represents the distance between amino acid i and amino acid j. Given z_ij=1 and z_jk=1, we can know that z_ik < 2. Thus we can determine the range of z_ik distance through z_ij and z_jk distances, meaning we can constrain possible values of z_ik through z_ij and z_jk. Therefore, triangular updates and triangular attention mechanisms are designed to encode such geometric constraints into the model.
  - So, the value of z_ij can be updated by obtaining all possible k to get (z_ik, z_jk). Since in reality, z_ij doesn't just contain distance information but represents relationship information between i and j, it's also directional. z_ij and z_ji have different meanings.
  - Based on graph computation theory, in triangular relationships, the directions of the other two edges used to update edge z_ij are divided into incoming and outgoing. For z_ij:

    ![image.png](../images/image%2039.png)

    - Its outgoing edges are: z_ik, z_jk
    - Its incoming edges are: z_ki, z_kj
  - Why distinguish between outgoing edges and incoming edges? Why can't they be mixed? → The current understanding is that two edges simultaneously point from i and j to k or simultaneously from k to i and j. Since edges are directional, simultaneously pointing from i and j to k or vice versa, these two edges have consistent physical meaning, which is the relationship of i and j to k (or vice versa), making it easier for the model to accurately model. (This understanding is still not thorough enough, will organize when there's an opportunity.)
- Next, let's see how triangular update and triangular attention are specifically calculated:

  - Triangular Update

    - Outgoing:

      - Specific algorithm implementation:

        ![image.png](../images/image%2040.png)

        1. The vector z_i_j performs LayerNorm on itself, i.e., normalization in the c_z dimension.
        2. z_i_j undergoes linear transformation to convert to a vector of dimension c=128, then calculates sigmoid value at each position, then element-wise multiplies with another linearly transformed vector to get a_i_j or b_i_j with dimension c.
        3. Then z_j_j first undergoes linear transformation (transformation dimension unchanged, still c_z), then calculates sigmoid to get g_i_j with dimension c_z, this vector is used for gating.
        4. Finally, calculate Triangular Update. To update z_i_j, we need to obtain from calculations of a_i_k and b_j_k (k has N_token choices). The specific method is: select the i-th row from {a_i_j} to get {a_i} (N_token vectors), select the j-th row from {b_i_j} to get {b_j} (N_token vectors), then for the k-th element in {a_i} and {b_j}, calculate element-wise multiplication to get a c-dimensional vector, then sum all N_token vectors to get a c-dimensional vector, then perform LayerNorm calculation, finally perform linear transformation to get a c_z-dimensional vector; finally element-wise multiply g_i_j to get the final z_i_j result.
      - Graphical explanation:

        ![image.png](../images/image%2041.png)
    - Incoming:

      - Specific algorithm implementation:

        ![image.png](../images/image%2042.png)

        - Note the main change here is calculating a_k_i and b_k_j, i.e., calculating from the column perspective, which is exactly symmetric to the previous calculation method. This can also be clearly seen from the diagram below.
      - Graphical explanation:

        ![image.png](../images/image%2043.png)
  - Triangular Attention

    - Triangular Attention (Starting Node corresponding to outgoing edges)
      - Specific algorithm implementation:

        ![image.png](../images/image%2044.png)

        1. First perform LayerNorm normalization on z_i_j.
        2. For the specific h-th head among N_head heads, perform different linear transformations on z_i_j to get q_h_i_j, k_h_i_j, v_h_i_j, with dimension transformations all being c_z → c.
        3. For the specific h-th head among N_head heads, perform linear transformation on z_i_j to get b_h_i_j. Dimension transformation is c_z → 1.
        4. For the specific h-th head among N_head heads, first perform linear transformation on z_i_j, dimension transformation c_z → c, then calculate sigmoid value for each element to get g_h_i_j for subsequent gating.
        5. Calculate Triangular Attention step 1: calculate attention score. For q_h_i_j and k_h_i_k, calculate dot product, then divide by sqrt(c), then add b_h_j_k (this 1-dimensional value), the resulting scalar then calculates softmax value in the k dimension to get attention score a_h_i_j_k at position k. This is a scalar value, which can also be understood as a weight value for subsequent multiplication with value.
        6. Calculate Triangular Attention step 2: calculate attention result at position (i,j), through weighted sum of a_h_i_j_k and v_h_i_k, get attention value at position (i,j), this is a c-dimensional vector; then element-wise multiply with g_h_i_j to get gated attention vector o_h_i_j, also with dimension c.
        7. Finally, for values at position {i,j}, merge multiple heads. First concatenate according to h heads in the last feature dimension, dimension change: c → h*c; then perform another linear transformation, changing dimension from h*c → c to get the final result z_i_j.
      - Graphical explanation:

        ![image.png](../images/image%2045.png)

        - Note that Triangular Attention here is actually a variant of Axial Attention, with added b_j_k as bias and added gating mechanism. But if we ignore these two added features, it's equivalent to doing self-attention row-wise.
    - Triangular Attention (Ending Node corresponding to incoming edges)
      - Specific algorithm implementation

        ![image.png](../images/image%2046.png)

        - The main difference here is in the calculation method of Triangular Attention:
          1. Use q_h_i_j and k_h_k_j to calculate attention score, then add bias b_h_k_i as the result of a_h_i_j_k. Note that q_h_i_j calculates dot product with k_h_k_j rather than k_h_k_i, and the added bias is b_h_k_i rather than b_h_k_j. I guess the reason is for convenience in calculating Axial Attention, otherwise it wouldn't be column-based self-attention. See the diagram below for details.
      - Graphical explanation

        ![image.png](../images/image%2047.png)

        - The original diagram was incorrect and didn't faithfully follow the implementation in the official documentation. The red box part corrected the original erroneous annotations in the diagram.
- Finally, let's see how Single Attention with pair bias is implemented.

  - The inputs are token-level single representation {s_i} and token-level pair representation {z_i_j}, and the output is {s_i}.
  - Mainly use {z_i_j} as bias, added to the self-attention calculation of {s_i}, while also adding gating mechanism to the self-attention of {s_i}.
  - The specific algorithm pseudocode implementation is:

    ![image.png](../images/image%2048.png)

    - single representation {s_i} undergoes normalization calculation.
    - Calculate q, k, v representations for specific h-th head from {s_i}: q_h_i, k_h_i, v_h_i all undergo linear transformation from c_token → c.
    - Calculate a bias value from pair representation {z_i_j}, ready to be applied in self-attention of {s_i}: first {z_i_j} undergoes normalization, then linear transformation, dimension from c_z → 1, to get b_h_i_j.
    - Use {s_i} to calculate values for subsequent gating: first perform linear transformation on {s_i} c_token → c, then calculate sigmoid value for each element to get g_h_i.
    - Then calculate attention, which is actually normal self-attention, except when calculating attention score, add bias value b_h_i_j from pair representation to get scalar A_h_i_j.
    - Then use v_h_j to multiply each j by A_h_i_j and sum to get a weighted sum vector, then element-wise multiply g_h_i to get attention result for specific h-th head, then concatenate the results in the last dimension and finally undergo linear transformation to get the result of {s_i} after attention, with dimension c_token.
  - The specific graphical explanation is as follows:

    ![image.png](../images/image%2049.png)