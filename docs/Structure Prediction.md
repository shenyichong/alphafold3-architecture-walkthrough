# Structure Prediction

## Basic Concepts of Diffusion

- In AlphaFold3, the entire structure prediction module adopts atom-level diffusion. Simply put, diffusion works specifically as follows:

  - Starting from the most realistic original data, suppose it's a real panda photo, then continuously add random noise to this photo, and train the model to predict what kind of noise was added.
  - The specific steps are as follows:
    - Training phase:
      - Noise addition process:
        1. Assume x(t=0) is the original data. In the first time step, add some noise to data point x(t=0) to get x(t=1).
        2. In the second time step, add noise to x(t=1) to get x(t=2).
        3. Continue repeating this process. After T steps, the data is completely covered by noise, becoming random noise x(t=T).
      - Model objective: Given a noise-contaminated data point x(t) and time step t, the model needs to predict how this data point transformed from the previous step x(t-1), i.e., the model needs to predict what noise was added from x(t-1) to x(t).
      - Loss function: The difference between the model's predicted noise and the actually added noise.
    - Prediction phase:
      - Denoising process:
        1. Start from pure random noise, x(t=T) is completely random noise.
        2. At each time step t, the model predicts the noise that should be removed in this step, then removes this noise to get x(t-1).
        3. Repeat this process, gradually moving from x(t=T) to x(t=0).
        4. Finally get a denoised data point that should look like training data.
- What is Conditional Diffusion?

  - In diffusion models, the model can also "control" the generated results based on certain input information, which is conditional diffusion.
  - So whether in training or prediction process, at each time step, the model's input should include:
    - Current data point x(t) at time t.
    - Current time step t.
    - Conditional information (such as protein properties and other information, here mainly referring to token-level and atom-level single and pair representations as conditional information).
  - Model output: Predicted noise added from x(t-1) to x(t) (training), or predicted noise that should be removed from x(t) to x(t-1) (inference).
- How is Diffusion applied in AlphaFold3?

  ![image.png](../images/image%2050.png)

  - In AlphaFold3, the original data used for denoising is a matrix x with dimensions [N_atoms, 3], where 3 represents the xyz coordinates of each atom.
  - During training, the model will add Gaussian noise at each step based on a correct atomic three-dimensional coordinate sequence x, until the coordinates become completely random.
  - Then during inference, the model starts from a completely random atomic three-dimensional coordinate sequence. At each time step, it first performs a data-augmentation operation, rotating and transforming the three-dimensional coordinates. The purpose is to achieve the function of Invariant Point Attention (IPA) in AF2, proving that three-dimensional coordinates after rotation and transformation are equivalent to each other. Then it adds some noise to the coordinates to try to generate some different generated data. Finally, it predicts the denoised result of the current time step as the starting point for the next step.

  ![image.png](../images/image%2051.png)

## Detailed Structure Prediction

![image.png](../images/image%2052.png)

### Detailed Sample Diffusion (Inference Process)

- The application of basic diffusion process in alphafold3, here refers to the specific algorithm flow of the diffusion model's inference process in alphafold3's inference process, from the initial state (completely random three-dimensional structure), then through step-by-step denoising, finally returning a denoised result (predicted three-dimensional structure).
- The specific algorithm pseudocode and analysis are shown as follows:

  ![image.png](../images/image%2053.png)

  - First, its input parameters include many inputs for subsequent conditional diffusion, including f*, {s_inputs_i}, {s_trunk_i}, {z_trunk_i_j}, which are mainly processed in the DiffusionModule algorithm later, and are temporarily ignored here.
  - Other inputs are mainly those that need attention in the diffusion algorithm, including Noise Schedule $[c_0, c_1, ..., c_T]$, scaling factors ($γ_0$ and $γ_{min}$), noise scaling coefficient noise_scale $λ$, and step scaling coefficient step_scale η. Their specific functions are introduced as follows:
    - Noise Schedule: Defines the noise intensity at each step of the diffusion process, with values ranging from [0,1]. It is a series of pre-set scalars. Generally, the noise intensity is maximum at t=0, then gradually decreases, ending at smaller t=T.
    - Scaling factors ($γ_0$ and $γ_{min}$), noise scaling coefficient noise_scale $λ$, are all used at the beginning of each step of Sample Diffusion inference, when noise needs to be added to the iteration result of the previous step first, generating noise $\hat\xi_l$.
    - Step scaling coefficient step scale η: Mainly used to control the update amplitude of each iteration step when x_l is updated later. In the process of x_l update $\vec{x}_l \leftarrow \vec{x}_l^{\text{noisy}} + \eta \cdot dt \cdot \vec{\delta}_l$: if η > 1, increase the update amplitude and accelerate the denoising process; if η < 1, reduce the update amplitude, making the denoising process smoother, but may require more iteration steps.
  - The specific algorithm flow analysis is as follows:
    1. Initially, $\vec{x}_l$ is completely random three-dimensional noise with dimension [3], and {$\vec{x}_l$} has dimension [N_atoms, 3]. Where $\mathcal{N}(\vec{0}, \mathbf{I}_3)$ is a multivariate normal distribution with mean as three-dimensional vector [0,0,0], indicating that the means of all three dimensions are 0; covariance matrix is [1, 0, 0; 0, 1, 0; 0, 0, 1], indicating that the dimensions are independent of each other and all have variance 1.
    2. Next, enter the loop for each time step, from $\tau=1$ to $\tau=T$:
    3. First perform data augmentation. The purpose here is to solve the problem that AlphaFold2 previously used Invariant Point Attention method to solve, namely solving rotational invariance and translational invariance. That is, the coordinates of a sequence's three-dimensional structure obtained through random rotation and translation are actually equivalent new coordinates, the three-dimensional structure essentially unchanged, and the relative positions between atoms unchanged.
    4. Generally, $c_\tau < \gamma_{\text{min}}$, because here $\gamma_{\text{min}} = 1$, so $\gamma=0$.
    5. So, the time step here is $\hat{t} = c_{\tau-1}$.
    6. The calculated added noise is: $\vec{\xi}_l = 0*(\vec{0}, \mathbf{I}_3)$, no noise is actually added.
    7. Thus we get the final noisy result $\vec{x}_l^{\text{noisy}} = \vec{x}_l$
    8. At this time, call DiffusionModule (detailed in the next stage) to calculate the real inference result of this step, obtaining the denoised result $\{\vec{x}_l^{\text{denoised}}\}$.
    9. Then start calculating the denoising direction vector $\vec{\delta}_l = \frac{\left( \vec{x}_l^{\text{noisy}} - \vec{x}_l^{\text{denoised}} \right)}{\hat{t}}$, i.e., the direction and magnitude from noisy coordinates $\vec{x}_l^{{noisy}}$ to denoised coordinates $\vec{x}_l^{\text{denoised}}$, then perform noise normalization to keep the denoising direction stable in the noise intensity changes of each time step. This can be understood as something like "gradient" or "directional derivative" in the diffusion process.
    10. Then calculate the time step difference dt, the difference between the current time step and the previous time step, providing a "step size" for the update, the difference from $c_\tau$ to the previous time step parameter $\hat{t}$, which is actually exactly $dt = c_\tau - c_{\tau-1}$.
    11. Finally, update $\vec{x}_l$, starting from the noisy coordinates $\vec{x}_l^{\text{noisy}}$, which is actually $\vec{x}_l=\vec{x}_l+\eta \cdot dt \cdot \vec{\delta}_l$. Since dt is likely a negative number here, it's actually subtracting noise.

### Detailed Diffusion Module (Inference Process)

![image.png](../images/image%2054.png)

- **DiffusionConditioning**: Prepare token-level conditioning tensors (pair representation z_i_j and single representation s_i)
- **AtomAttentionEncoder**: Prepare atom-level conditioning tensors (pair representation p_l_m, single representation q_l, c_l), and use them to generate token-level single representation a_i.
- **DiffusionTransformers**: token-level single representation a_i undergoes attention calculation, then maps back to atom-level.
- **AtomAttentionDecoder**: Perform attention calculation at atom-level to get predicted atom-level denoising results.

Note 1: Here, atom-level attention is all labeled by the author as local attention, and token-level attention is all labeled as global attention. The reason is that the number of atoms is very large. When calculating attention between atomic sequences, i.e., when calculating AtomTransformer, sparse attention is actually calculated. Other atoms that are too far from the current query atom do not participate in the current atom's attention calculation, otherwise the computational cost would be very large, so it's called local attention. When calculating token-level attention, all global token information is considered, so it's called global attention.

Note 2: The 3 blocks in AtomAttentionEncoder and 3 blocks in AtomAttentionDecoder here refer to AtomTransformer, which is essentially DiffusionTransformer at atomic granularity with sparse bias added, while DiffusionTransformer at token granularity has 24 blocks.

1. **DiffusionConditioning**

   - Algorithm pseudocode is as follows:

     ![image.png](../images/image%2055.png)
   - Constructing token-level pair conditioning input: {z_i_j}

     ![image.png](../images/image%2056.png)

     1. First use f* to calculate relative position encoding. This relative position encoding is a function of (i,j), representing the relative position relationship between any two tokens. The resulting dimension is c_z; concatenate with z_trunk_i_j (dimension is also c_z) to get z_i_j with dimension 2*c_z.
     2. Then perform layerNorm on z_i_j, then perform linear transformation to transform to c_z dimension.
     3. Finally, after two Transition Layer additions, get a new z_i_j.
   - Constructing token-level single conditioning input: {s_i}

     ![image.png](../images/image%2057.png)

     1. First concatenate s_trunk_i and s_inputs_i, these two single representations, to get s_i with dimension becoming 2*c_s+65 (s_inputs_i has dimension c_s+65).
     2. Then normalize s_i, then perform linear transformation with dimension becoming c_s.
     3. Then map the diffusion time step information (scalar, specifically the noise schedule value of the current time step) to high-dimensional vector space to enhance the model's ability to capture nonlinear features of time steps.
        1. The specific pseudocode is as follows:

           ![image.png](../images/image%2058.png)

           1. Generate c-dimensional vectors, each dimension is independently normally distributed, to get w and b.
           2. Generate high-dimensional vector features of time steps. Through cos function, encode scalar time step t into a high-dimensional space. The specific generated vector can be understood as shown in the figure below (x is time step t, different values of y are its vectors in high-dimensional space). Each t cross-section is a high-dimensional vector at time t.

              ![image.png](../images/image%2059.png)

              - Different frequencies capture multi-scale features of time steps (low frequency represents global dynamics, high frequency represents local details)
              - Offset increases the diversity of embeddings, enabling the model to learn more complex temporal features.
        2. First normalize the high-dimensional time step information, then perform linear transformation (n → c_s), and add it to the token-level single representation s_i.
        3. After two more Transition layer additions, get a new s_i with dimension c_s.
   - By adding diffusion time step information in this DiffusionCondition part, the model can know the current time step of the diffusion process during the de-noising process and predict the correct scale of noise that needs to be removed.
   - The result after DiffusionCondition is information at token-level scale. Next, we need to calculate atomic-level information at atom-level.
2. **AtomAttentionEncoder**

   - First scale x_noisy_l, converting it to unit vectors with unit variance 1. The scaled result is a dimensionless value, which helps maintain numerical stability.

     ![image.png](../images/image%2060.png)
   - Then formally enter the AtomAttentionEncoder function for calculation:

     ![image.png](../images/image%2061.png)

     - AtomAttentionEncoder inputs include: {f*} (original feature reference conformation features), {r_noisy_l} (current time step atomic coordinates with current noise added), {s_trunk_i} (token-level single representation after Pairformer), {z_i_j} (token-level Pair representation after DiffusionConditioning)
     - AtomAttentionEncoder outputs include: a_i (token-level single representation), q_l (atom-level single representation calculated by this module), c_l (atom-level initial representation obtained based on reference conformation), p_l_m (atom-level pair representation calculated by this module).
     - AtomAttentionEncoder pseudocode is as follows: (diffusion newly added parts are highlighted)

       ![image.png](../images/image%2062.png)

       - First, calculate c_l from the original reference conformation representation, set the initial value of q_l to c_l, then calculate atom-level pair representation p_l_m from the reference conformation representation.
       - Next, when r_l is not empty (i.e., during the current diffusion part inference process):
         - Use s_trunk, this token-level single feature, get the token index tok_idx(l) corresponding to atom index l, then get the s_trunk vector corresponding to this token index with dimension c_s(c_token), then after LayerNorm, perform linear transformation with dimension from c_s → c_atom. Then add c_l itself to get new c_l. The specific process is as follows:

           ![image.png](../images/image%2063.png)
         - Use z, this token-level pair feature, get the token indices tok_idx(l) and tok_idx(m) corresponding to atom indices l and m, then get the z vector corresponding to these two dimension indices with dimension c_z, then after LayerNorm, perform linear transformation with dimension from c_z → c_atompair. Finally add p_l_m itself to get new p_l_m. The specific process is as follows:

           ![image.png](../images/image%2064.png)
         - For the current time step atomic coordinates with current noise added r_noisy_l, perform linear transformation with dimension transformation 3→c_atom, add to q_l to get the latest q_l result.
       - Finally, based on c_l, update the p_l_m result. Pass p_l_m through 3-layer MLP to get action p_l_m, then calculate the latest q_l through AtomTransformer. Finally, average q_l, this atom-level single representation, across different token dimensions to get a_i (token-level single representation) result. Thus, through AtomAttentionEncoder, we get the following results:
         - {q_l}: Updated atom-level single representation, containing current atom coordinate information.
         - {c_l}: atom-level single representation, a variable updated based on Pairformer's token-level single representation, mainly serving the role of conditioning based on Trunk.
         - {p_l_m}: atom-level pair representation, used for subsequent diffusion conditioning.
         - {a_i}: token-level single representation, aggregated from q_l. Contains both atom-level coordinate information and token-level sequence information.
3. **DiffusionTransformers**

   - The specific pseudocode is as follows: This part mainly performs self-attention on the token-level information a_i calculated in the previous step (which contains atomic three-dimensional coordinate information and sequence information).

     ![image.png](../images/image%2065.png)

     - First, starting from the token-level single representation {s_i} calculated from **DiffusionConditioning**, calculate its LayerNorm result, then perform linear transformation to transform it to the dimension of a_i, linear transformation dimension is c_token → c_s, then add {a_i} itself for element-wise add to get new {a_i}.
     - Then, perform attention on token-level information {a_i}, and use {s_i} and {z_i_j} calculated from DiffusionConditioning for conditioning. Note that a major difference between this DiffusionTransformer and all previous DiffusionTransformers is that this is for token-level (Token-level equivalent of the atom transformer), so here $\beta_{ij} = 0$ means no sparse attention bias is added. A schematic diagram is also given below:

       ![image.png](../images/image%2066.png)
     - Finally, a_i undergoes LayerNorm and outputs with dimension c_token.
4. **AtomAttentionDecoder**

   Pseudocode is as follows:

   ![image.png](../images/bed1bf65-6bec-4b73-acf5-b2cba14de665.png)

   - Now, we return to Atom space, using the updated a_i to broadcast it to each atom to update atom-level single representation q_l.

     ![image.png](../images/image%2067.png)
   - Then, use Atom Transformer to update q_l.

     ![image.png](../images/image%2068.png)
   - Then, after LayerNorm and linear transformation of the updated q_l, map it to the three-dimensional coordinates of the atomic sequence to get r_update_l.

     ![image.png](../images/image%2069.png)
   - Finally, outside AtomAttentionDecoder, rescale the "dimensionless" r_update_l to non-unit standard deviation x_out_l, and what's returned is x_denoised_l.

     ![image.png](../images/image%2070.png)