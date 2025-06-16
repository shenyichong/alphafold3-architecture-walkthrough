# Input Preparation

## **MSA和Templates是如何来的？**

### 为什么需要MSA？
- 存在于不同物种之间不同版本某一类蛋白，其序列和结构可能是相似的，通过将这些蛋白集合起来我们可以看到一个蛋白质的某些位置是如何随着进化改变的。
- MSA的每一行代表来自不同物种的相似蛋白。列上的保守模式可以反应出该位置需要某些特定氨基酸的重要性。不同列之间的关系则反应出氨基酸之间的相互关系（如果两个氨基酸在物理上相互作用，则他们在进化工程中氨基酸的变化也可能是相关的。）所以MSA经常用来丰富单一蛋白质的representation。

### 为什么需要Template：

- 类似的，如果上述MSA中包含了已知的结构，那么那么它很有可能能够帮助预测当前蛋白质的结构。Template只关心单链的结构。

### 如何获取MSA？

- 使用genetic search搜索相近的蛋白质或者RNA链，一条链一般搜索出N_msa（<16384）条相近的链：
    
    ![image.png](../images/image.png)

- 如果是多条链，如果来自相同物种可以配对（paired），那么MSA的矩阵可能的形式是这样的：
        
    ![image.png](../images/image%201.png)
        
- 否则是这样的，构成一个diagnoal matrix：
        
    ![image.png](../images/image%202.png)
        
### 如何获取Template？

- 使用Template Search，对于生成的MSA，使用HMM去PDB数据库中寻找与之相似的蛋白质序列，接下来从这些匹配到的序列中挑选4个最高质量的结构作为模版。

### 如何表征Templates？

- 计算每一个token和token之间的欧几里得距离，并且是用离散化的距离表示（具体来说，值被划分为38个区间，范围从3.15Å到50.75Å，外加一个额外的区间表示超过50.75Å的距离）。
- 如果某些token包含多个原子，则选择中心原子用于计算距离，比如 Cɑ 是氨基酸的中心原子，C1是核苷酸的中心原子。
- 模版只包含同一条链上的距离信息，忽略了链之间的相互作用。

## **如何构建Atom-level的表征？**

- 构建得到p和q：【对应Algorithm 5 AtomAttentionEncoder】
    - 为了构建atom-level的single representation，首先需要所有的原子层面的特征，第一步是首先对每一个氨基酸、核苷酸以及ligand(配体)构建一个参考构象(reference conformer)，参考构象可以通过特定的方式查出来或者算出来，作为局部的先验三维结构信息。
    - c 就是针对参考构象中的各个特征进行concate之后，再进行一次线性变换之后的结果。c的形状为[C_atom=128, N_atoms]，其中C_atom为每一个atom线性变换之后的特征维度，N_atoms为序列中所有原子的个数。c_l代表l位置的原子的特征，c_m代表m位置的原子的特征。
        
        ![image.png](../images/image%203.png)
        
    - atom-level的single representation q，通过c进行初始化：
        
        ![image.png](../images/image%204.png)
        
    - 然后使用c来初始化atom-level的原子对表征p（atom-level pair representation），p表征的是原子之间的相对距离，具体过程如下：
        1. 计算原子参考三维坐标之间的距离，得到的结果是一个三维的向量。
            
            ![image.png](../images/image%205.png)
            
        2. 因为原子的参考三维坐标仅仅是针对其自身参考构象得到的，是一个局部信息，所以仅仅在token内部（氨基酸、核苷酸、配体等）计算距离，所以需要计算得到一个mask v，只有相同chain_id和residue_idx的情况下才相互之间计算坐标位置差异，这时候v为1，其他时候v为0
            
            ![image.png](../images/image%206.png)
            
        3. 计算p，形状为[N_atoms, N_atoms, C_atompair=16]: 
            
            ![image.png](../images/image%207.png)
            
            - p(l,m) 向量的维度为[C_atompair]，计算方式为，对d(l,m)三维向量通过一个线性层，得到一个维度为C_atompair的向量，同时乘一个用于mask的标量v(l,m)。
            - inverse square of the distances 1/(1+||d(l,m)||^2)是一个标量，然后经过一个线性变换，变成[C_atompair]的向量，同时还是乘一个用于mask的标量v(l,m)。p(l,m) = p(l,m) + 这个新的向量。
            - 最后，p(l,m)再加上mask这个标量。
            
            ![image.png](../images/image%208.png)
            
            - p(l,m) 还需要加上原始的c中的信息，包含c(:, l)的信息和c(:, m)的信息，这两个信息都先经过relu然后再做线性变换，变换成C_atompair的向量，加到p(l,m)中。
            - 最后p(l,m) = p(l,m) + 三层MLP(p(l,m))
- 更新q（Atom Transformer）：
    1. Adaptive LayerNorm:【对应Algorithm 26 AdaLN】
        - 输入为c和q，形状都是[C_atom, N_atoms]，c作为次要输入，主要用于计算q的gamma和beta的值，这样通过c来动态调整q的layerNorm的结果。
        - 具体的来说，正常的LayerNorm是这样做的：
            
            ![image.png](../images/image%209.png)
            
        - 然后Adaptive Layer Norm是这样做的：
            
            ![image.png](../images/image%2010.png)
            
            - 公式是这样的：
                
                ![image.png](../images/image%2011.png)
                
                - 这里的a就是q，s就是c。
                - 计算的时候sigmoid(Linear(s))相当于就是新的gamma，LinearNoBias(s)相当于就是新的beta。
    2. Attention with Pair Bias 【对应Algorithm 24 AttentionPairBias】:
        - 输入为q和p，q的形状为[C_atom, N_atoms]，p的形状是[N_atoms, N_atoms, C_atompair]。
        - 作为典型的Attention结构，(Q, K, V)都来自于q，形状为[C_atom, N_atoms]。
            - 假设是N_head头的attention，其中a_i代表q中的第i个原子的向量q_i,那么对于第h个头，第i个q向量，得到其(q_h_i, k_h_i, v_h_i):
                
                ![image.png](../images/image%2012.png)
                
                - 这里的维度c是这样得到的：
                    
                    ![image.png](../images/image%2013.png)
                    
            - Pair-biasing：从哪里来？从p中提取第i行，即第i个原子和其他原子的关系，那么p_i_j就是第i个原子和第j个原子之间的关系，向量形状为[C_atompair]，在公式中使用z_i_j代表p_i_j。
                - z_i_j在C_atompair维度上先做了一次LayerNorm，然后再做一次线性变换，从C_atompair维降到1维：
                    
                    ![image.png](../images/image%2014.png)
                    
                - 然后计算softmax之前引入此pair Bias，此时针对第i个原子和第j个原子的向量q_h_i和k_h_i先做向量点乘在scale之后，再加上一个标量b_h_i_j后进行softmax，得到权重A_h_i_j：
                    
                    ![image.png](../images/image%2015.png)
                    
                - 然后再直接计算对于第i个原子的attention结果：
                    
                    ![image.png](../images/image%2016.png)
                    
            - Gating：从q中获取第i个原子的向量：
                - 先做一次线性变换，从c_atom维变到Head的维度c上，然后再直接求sigmoid，将其映射到0到1之间到一个数上作为gating。
                    
                    ![image.png](../images/image%2017.png)
                    
                - 然后element-wise乘以attention的结果，最终将所有的Head concate起来，并最后经过一个线性变换，得到attention的结果q_i，形状是[C_atom]。
                    
                    ![image.png](../images/image%2018.png)
                    
            - Sparse attention：因为原子的数量远远大于tokens的数量，所以这里计算atom attention的时候，从计算量上来考虑，并不会计算一个原子对所有原子的attention，而是会计算一个局部的attention，叫做sequence-local atom attention，具体的方法是：
                - 在计算attention的时候，将不关心位置(i,j)的softmax的结果做到近似于0，那么相当于在softmax之前特定位置(i,j)的值为负无穷大，通过引入的beta_i_j来进行区分：
                    
                    ![image.png](../images/image%2019.png)
                    
                    - 如果i和j之间的距离满足条件，那么就是需要计算atom attention的，这时候beta_i_j为0。
                    - 如果i和j之间距离不满足条件，那么就是不需要计算attention的，这时候beta_i_j为-10^10。
    3. Conditioned Gating: 【对应Algorithm 24 AttentionPairBias】
        - 输入是c和q，形状分别是[C_atom, N_atoms], [C_atom, N_atoms]，输出是q，形状是 [C_atom, N_atoms]。
        - 这里公式中s_i实际上就是c_i ，先做一次线性变换，然后在计算sigmoid，将c_i中的每一个元素映射到0和1之间，最后在element-wise乘以a_i，实际上就是q_i, 得到最新的q_i:
            
            ![image.png](../images/image%2020.png)
            
    4. Conditioned Transition: 【对应Algorithm 25 ConditionalTransitionBlock】
        - 输入是c，q和p，形状分别是[C_atom, N_atoms], [C_atom, N_atoms], [N_atoms, N_atoms, C_atompair]，输出是q，形状是[C_atom, N_atoms]。
        - Atom Transformer的最后一个模块，这个相当于在Transformer中的MLP层，说他是Conditional是因为他包含在Adaptive LayerNorm层（前面第三步）和Conditional Gating（前面第三步）之间。区别只是中间的是MLP还是Attention。
            - 第一步还是一个Adaptive LayerNorm：
                
                ![image.png](../images/image%2021.png)
                
            - 第二步是一个swish：
                
                ![image.png](../images/image%2022.png)
                
            - 第三步是Conditional Gating：
                
                ![image.png](../images/image%2023.png)
                

## **如何构建Token-level的表征？**

- 构建Token-level的Single Sequence Representation
    - 输入是q，atom-level single representation，形状是[C_atom, N_atoms]。输出是S_inputs和S_init，形状分别是[C_token+65, N_tokens]，[C_token, N_tokens]。
    - 首先将每一个atom的表示维度C_atom经过线性变换到C_token，然后经过relu激活函数，然后在相同的token内部，对所有的atom表示求平均，得到N_token个C_token维度的向量。【来自 Algorithm 5 AtomAttentionEncoder】
        
        ![image.png](../images/image%2024.png)
        
    - 然后针对有MSA特征的token，拼接上residue_type（32）和MSA特征（MSA properties：32+1），从而得到S_input。【来自 Algorithm 2 InputFeatureEmbedder】
        
        ![image.png](../images/image%2025.png)
        
    - 最后再进行一次线性变换，从S_input转换为S_init。【来自 Algorithm 1 MainInferenceLoop】
        
        ![image.png](../images/image%2026.png)
        
        - 注意：这里的C_s就是C_token = 384
- 构建Token-level的Pair Representation 【来自 Algorithm 1 MainInferenceLoop】
    - 输入是S_init，形状是[C_token=384, N_tokens]，输出是Z_init，形状是[N_tokens, N_tokens, C_z=128]。
    - 要计算Z_init_i_j，那么需要得到特定两个token的特征，将其分别进行线性变换之后，再相加得到第一个z_i_j，向量长度从C_tokens也就是C_s=384转换为C_z=128。
        
        ![image.png](../images/image%2027.png)
        
    - 然后在z的（i，j）位置加入相对位置编码：
        
        ![image.png](../images/image%2028.png)
        
        - 详解RelativePositionEncoding：注意这里的i和j都是指的token。【来自 Algorithm 3 RelativePositionEncoding】
            
            ![image.png](../images/image%2029.png)
            
            - a_residue_i_j：残基相对位置信息：
                - 如果i和j token在同一个链中，那么d_residue_i_j 则是i残基和j残基相对位置之差，范围在[0, 65]。
                - 如果i和j token不在同一个链中，那么d_residue_i_j = 2*r_max+1 = 65
                - a_residue_i_j 为一个长度为66的one-hot编码，1的位置为d_residue_i_j的值。
            - a_token_i_j：token相对位置信息：
                - 如果i和j token在同一个残基中，（对于modified 氨基酸或者核苷酸，那么一个原子是一个token），那么d_token_i_j 为不同残基序号之差，范围在[0, 65]。
                - 如果i和j token不在同一个残基中，取最大值d_token_i_j = 2*r_max+1 = 65
                - a_token_i_j 为一个长度为66的one-hot编码，1的位置为d_token_i_j的值。
            - a_chain_i_j：链相对位置信息：
                - 如果i和j token不在同一条链中，那么d_chain_i_j 为链之间的序号之差，范围在[0, 5]。
                - 如果i和j token在同一条链中，那么d_chain_i_j设置为最大值5。
                - a_chain_i_j为一个长度为6的one-hot编码，1的位置为d_chain_i_j的值。
            - b_same_entity_i_j：如果i和j token在同一个实体（完全相同的氨基酸序列为唯一实体，拥有唯一实体id）中，为1否则为0.
            - 最后将[a_residue_i_j, a_token_i_j, b_same_entity_i_j, a_chain_i_j] 拼接起来，得到了一个长度为66+66+1+6 = 139 = C_rpe 的向量。
            - 然后再经过一次线性变换，将其向量维度变换到C_z=128维度。
    - 最后在将token的bond信息加入，通过线性变换之后，加入z_i_j，得到最后的结果。
        
        ![image.png](../images/image%2030.png)