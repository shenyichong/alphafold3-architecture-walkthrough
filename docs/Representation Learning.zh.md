# Representation Learning

## Template Module

- Template Module的输入是什么？
    - token-level pair representation { z_i_j } 和 包含template信息的特征 {**f**} 。
- Template 特征如何构建？【来自 Algorithm 16 TemplateEmbedder】
    
    ![image.png](../images/image%2031.png)
    
    1. 针对template t，b_template_backbone_frame_mask_i_j 通过对第t个模版的i个位置和j个位置的template_backbone_frame_mask的值来计算，即只有i和j位置都包含了可以用于计算backbone的所有原子，此mask结果为1，否则为0.
    2. 同理，对第t个模版的i位置和j位置的template_pseudo_beta_mask都为1，此b_template_pseudo_beta_mask_i_j这个mask才为1，否则为0。含义是中心原子在i和j位置是否有坐标。
    3. a_t_i_j 计算第t个模版的i位置和j位置的相关特征，将template_distogram （表示两个token之间距离，形状为[N_templ, N_token, N_token, 39]）的最后一维和b_template_backbone_frame_mask_i_j（1维标量），template_unit_vector（表示两个token之间包含方向的单位向量，形状为[N_templ, N_token, N_token, 3]）的最后一维，和b_template_pseudo_beta_mask_i_j (1维标量) 拼接起来成为一个维度为44的向量。
    4. 如果位于i位置的token的和j位置的token不在一条链上，则a_t_i_j 向量设置为全0向量。
    5. a_t_i_j再拼接上位于t模版的i位置的残基类型(one-hot encoding) 和j位置的残基类型信息，得到a_t_i_j的向量为一个[44+32+32 =108]的向量。{a_t_i_j}的形状为[N_templ, N_token, N_token, 108]。
- 然后针对每一个template，做相同以下操作：【来自 Algorithm 16 TemplateEmbedder】
    
    ![image.png](../images/image%2032.png)
    
    1. 首先定义一个u_i_j用于存放结算结果，{u_i_j}的形状是[N_token, N_token, c]
    2. 针对每一个template，进行相同的操作：对a_t_i_j进行线性变换，从108维 → c维， 对z_i_j先进行LN，然后再进行线性变换，从c_z维 → c维，然后针对每一个template的a_t_i_j，都加上相同的z_i_j变换后的结果，得到v_i_j，维度为c维。
    3. 然后将v_i_j通过PairformerStack进行计算，得到的结果再加上v_i_j 作为v_i_j的结果，维度为c维。
    4. 最后对v_i_j 进行LN之后，对于所有的template都加到u_i_j向量上，维度为c维。
- 最后对u_i_j的结果进行平均，除以template的数量，最后再激活一下，然后经过一次线性变换得到最终的结果u_i_j， {u_i_j}的形状为[N_token, N_token, c] 。【来自 Algorithm 16 TemplateEmbedder】
    
    ![image.png](../images/image%2033.png)
    

## MSA Module

![image.png](../images/image%2034.png)

- 此模块的目标：同时更新MSA和Pair Representation特征，并让其相互影响。先通过Outer Product Mean来让MSA表征更新Pair Representation，然后再使用Pair Representation通过 row-wise gated self-attention using only pair bias来更新MSA的表征。最终Pair representation经过一系列的三角计算并最终进行更新。
- 输入：
    - MSA 表征（会按照行进行subsamle，只随机选取其中少部分样本）
        - f_msa: 形状为 [N_msa, N_token, 32]，原始的N_msa条MSA样本，每一个位置有32种可能性。
        - f_has_deletion: 形状为 [N_msa, N_token], 原始信息，指示每个位置左边是否有删除。
        - f_deletion_value: 形状为 [N_msa, N_token]，原始信息，指示Raw deletion counts，大小在[0,1]之间。
    - S_inputs_i : 初始的token-level single sequence特征，形状为[N_token, C_token+65]，这里的65是32+32+1，其中包含了当前token的restype，以及MSA的一些特征，包括在i位置的各个restype类型的分布，以及在i位置的deletion mean的值。如果当前token没有MSA信息，则这65维都为0.
    - token-level Pair Representation：{z_ij} ，形状为[N_token, N_token, C_z=128]
- 输出：更新之后的token-level Pair Representation：{z_ij} ，形状为[N_token, N_token, C_z=128]
- 直接看伪代码：
    
    ![image.png](../images/image%2035.png)
    
    1. 首先，通过对f_msa_Si，f_has_deletion_Si以及f_deletion_Si进行拼接，得到一个32+1+1=34的向量：**m_Si**。注意这里的S和i都分别是下标，S表示在N_msa中第S行（第S个样本），i表示每一行中第i个位置。
    2. 然后通过SampleRandomWithoutReplacement函数对N_msa进行subsample，{S}代表所有N_msa的可能的序号的集合，而{s}代表下采样之后的可能的序号的集合。
    3. 然后**m_si**代表的是针对下采样之后的样本的第s个样本第i个位置，然后进行线性变换，将维度从34→C_m=64。
    4. 然后加上{S_inputs_i} 进行线性变换后的结果得到一个新的MSA特征 **m_si**。这里S_inputs_i也包含了MSA在每一个位置上的相关特征（如果有的话），维度是 C_token+65。
    5. 然后针对N_block个block进行循环：
        1. 对MSA信息m_si进行OuterProductMean计算并融合到pair representation中：
            1. 输入{m_si}的形状是[n_msa, N_tokens, C_m]，具体的算法如下：
                
                ![image.png](../images/image%2036.png)
                
                1. 首先，对m_si进行LN，{m_si}的形状为[n_msa, N_tokens, C_m]
                2. 然后进行线性变换，得到a_si和b_si，线性变换是从C_m=64 → c=32
                3. 对于每一个s，计算a_si和b_sj的外积，相当于计算对于s这个MSA样本来说，计算其i位置和j位置的关系，通过外积来计算。
                    1. 长度为c的向量与长度为c的向量的外积的计算得到一个[c , c]矩阵，
                    2. 然后将s个[c, c]的矩阵按每个位置(i,j)求平均，
                    3. 最后将这个矩阵变成一个一维向量，得到o_ij，形状为c*c。
                4. 最后在经过一次线性变换，将c*c → c_z=128 维度，得到最终的z_ij，即通过MSA计算得到位置之间的信息。
                
                注1：这里通过OuterProductMean方法，将MSA的表征融合到pair representation中去。对于同一个MSA序列，通过外积的方式得到其任意两个位置之间的关系，然后对所有的MSA序列的这两个位置的结果求平均，得到任意两个位置之间在进化上的关系信息并融合到pair representation中去。
                
                注2：注意到这里是仅仅在进化序列内部进行计算，然后进化序列之间的信息是唯一一次通过平均的方式融合在一起，避免了AF2中进化序列之间复杂的计算。
                
        2. 使用更新后的pair representation和m_si来更新m_si(MSA特征): MSA row-wise gated self-attention using only pair bias
            1. 输入是{m_si}的形状是[n_msa, N_tokens, C_m=64]， {z_ij}的形状是[N_token, N_token, C_z=128]。输出是{m_si}，形状是[n_msa, N_tokens, C_m=64]。具体的算法如下：
                
                ![image.png](../images/image%2037.png)
                
                1. 首先对MSA特征m_si进行LN。
                2. 然后对m_si进行多头线性变换，得到H_head个头v_h_si，线性变换维度从C_m→ c。
                3. 对pair represnetation特征z_ij首先进行LN，然后进行多头线性变换，得到b_h_ij，维度从C_z→1。
                4. 对m_si进行多头线性变换，维度从C_m→c，然后计算其sigmoid的值，得到g_h_si，用于后续做gating。
                5. 对b_h_ij沿着j的方向做softmax，得到weight w_h_ij，维度为1。
                6. 这里比较难理解的是如何得到o_h_si，这里v_h_sj和w_h_ij在j方向上按元素相乘，然后加起来得到o_h_si的中间结果。
                    1. 即通过对于{w_h_ij}这个形状为[N_token, N_token]的矩阵，取其第i行的元素[N_token]。
                    2. 对{v_h_sj}这个形状为[n_msa, N_token, c]的矩阵取其第s行的元素[N_token, c]。
                    3. 将其进行按元素相乘并加总起来，得到一个c维度的向量，其的位置为原来{m_si}矩阵的s行和i列。
                    4. o_h_si 再在c维度上按元素乘以g_h_si，进行gating。
                7. 最后针对o_h_si，将其H_head个头concate起来，得到一个c*H_head长的向量，然后经过线性变换后得到最终结果m^_si。
            2. 注意，这部分是通过pair representation来更新MSA的表征，更新方式是对每一个MSA的序列来说，其更新是相互独立的。然后使用pair representation中的位置之间的关系来构建weight，相当于给m_si中的每一个位置做了一次self-attention，引入了pair representation中的信息。
        3. {m_si} 再经过一层transition层后再作为下一个block的{m_si}的输入。
        4. pair representation {z_ij}经过一系列的三角计算和transition后，再作为下一个block{m_si}的输入。

## Pairformer Module

![image.png](../images/image%2038.png)

- 首先了解Pairformer这部分模块主要做了什么事情：Pair Representation会进行三角更新和三角注意力计算，并且用于更新Single Representation。和AF2不同的是，这里Single Representation不会去影响Pair Representation。
- 输入输出：输入是token-level pair representation {z_ij} 和 token-level single representation {s_i} ，他们的形状分别是[N_token, N_token, C_z=128] 和 [N_token, C_s=C_token=384]。
- 为什么要关注三角关系？（Why look at Triangles？）
    - 三角不等式说：三角形任意两边之和大于第三边。而在pair representation 中表征了任意两个token之间的关系，为了简化理解，我们可以将其看作是一个序列中任意两个氨基酸之间的距离，那么z_ij代表i氨基酸和j氨基酸之间的距离，那么已知z_ij=1, z_jk=1，那么我们就可以知道z_ik < 2。这样我们可以通过z_ij和z_jk的距离来确定z_ik的距离的范围，也就是说可以通过z_ij和z_jk来约束z_ik可能的值，故三角更新和三角注意力机制就是为了将这种几何约束编码到模型之中。
    - 所以，z_ij的值可以通过获取所有可能的k得到的（z_ik，z_jk）来进行更新，由于真实情况是，z_ij并不仅仅包含了距离信息，它代表了i和j之间的关系信息，因此它也是有方向的，z_ij和z_ji表示的含义是不一样的。
    - 并且基于图（graph）计算理论，将三角关系中，用于更新z_ij这一条边的另外两边的方向分为incoming和outgoing。则针对z_ij，
        
        ![image.png](../images/image%2039.png)
        
        - 其outgoing edges是：z_ik, z_jk
        - 其incoming edges是：z_ki, z_kj
    - 为什么要区分outgoing edges和incoming edges？为什么不能混用？ → 当前暂时的理解是，两条边同时从i和j指向k或者同时从k指向i和j，因为edge是有方向性的，同时从i和j指向k或相反，这两条边的物理含义是一致的，就是i和j对k的关系（或相反），更加便于模型准确地建模。（这里的理解还是不够透彻，等有机会再梳理。）
- 接下来看看具体是如何计算trianglar update和triangular attention是如何计算的：
    - Triangular Update
        - Outgoing：
            - 具体的算法实现：
                
                ![image.png](../images/image%2040.png)
                
                1. z_i_j 这个向量，自己进行LayerNorm，即在c_z维度上进行归一化。
                2. z_i_j进行线性变换，转换为维度为c=128的向量，然后每一个位置计算sigmoid的值，然后和另外一个进行了线性变换的向量进行按元素相乘，得到一个a_i_j 或者b_i_j，维度是c维。
                3. 然后还是对z_j_j先进行线性变换（变换维度不变，还是c_z），然后计算sigmoid，得到g_i_j，维度是c_z，这个向量用于gating。
                4. 最后，计算Triangular Update，要更新z_i_j，那么需要从a_i_k和b_j_k的计算中得到（k有N_token个选择）。具体方法是：那么从{a_i_j}中选取i行，得到{a_i}（有N_token个向量），从{b_i_j}中选取j行，得{b_j}（有N_token个向量），然后对{a_i}和{b_j}中的第k个元素，计算按元素相乘，得到一个c维的向量，然后再将所有的N_token个向量加起来，得到一个c维度的向量，然后进行LayerNorm计算，最后进行线性变换得到一个c_z维度的向量；最后按照元素相乘g_i_j，得到最终的z_i_j的结果。
            - 图示化解释：
                
                ![image.png](../images/image%2041.png)
                
        - Incoming：
            - 具体的算法实现：
                
                ![image.png](../images/image%2042.png)
                
                - 注意这里的主要变化，就是计算的是a_k_i和b_k_j，即从列的角度进行计算，和前面的计算方法刚好对称。从下面的图示也可以明显的看出来。
            - 图示化解释：
                
                ![image.png](../images/image%2043.png)
                
    - Triangular Attention
        - Triangular Attention (Starting Node 对应outgoing edges)
            - 具体算法实现：
                
                ![image.png](../images/image%2044.png)
                
                1. 首先对z_i_j 进行LayerNorm归一化处理。
                2. 针对N_head个头的特定h头，对z_i_j进行不同的线性变换，得到q_h_i_j, k_h_i_j, v_h_i_j，维度变换均为c_z → c。
                3. 针对N_head个头的特定h头，对z_i_j进行线性变换，得到b_h_i_j。维度变换为c_z → 1。 
                4. 针对N_head个头的特定h头，对z_i_j先进行线性变换，维度变换为c_z → c，然后每一个元素计算sigmoid的值，得到g_h_i_j用于后续的gating。
                5. 计算Triangular Attention 第一步：计算attention score，针对q_h_i_j和k_h_i_k计算点积，然后再除以sqrt(c)，然后再加上b_h_j_k这个维度为1的值，得到的一个标量再在k维度上计算softmax的值，得到在k位置上的attention score a_h_i_j_k。这是一个标量值，页可以理解是一个weight的值，用于后续乘以value。
                6. 计算Tiangular Attention 第二步：计算(i,j)位置上的attention的结果，通过a_h_i_j_k 与 v_h_i_k 的weighted sum，得到attention在(i,j)位置上的值，这是一个维度为c的向量；然后与g_h_i_j进行按元素相乘，得到gating之后的attention向量o_h_i_j，维度也为c。
                7. 最后针对{i,j}位置的值，将多头进行合并，首先按照h个多头，在最后一个特征维度进行拼接，维度变化：c → h*c；然后再进行一次线性变换，将维度从 h*c → c 得到最终的结果z_i_j。
            - 图示化解释：
                
                ![image.png](../images/image%2045.png)
                
                - 这里注意，其实这里的Triangular Attention就是一种Axial Attention的变种，增加了b_j_k作为bias，增加了gating的机制。但是如果抛开这两个增加到特性，相当于是按行在做self-attention。
        - Triangular Attention (Ending Node 对应incoming edges)
            - 具体算法实现
                
                ![image.png](../images/image%2046.png)
                
                - 这里的主要区别在计算Triangular Attention的方法：
                    1. 使用q_h_i_j 和 k_h_k_j 来计算attention score，然后再加上b_h_k_i 这个偏置，作为a_h_i_j_k的结果。这里需要注意的是q_h_i_j是和k_h_k_j而不是和k_h_k_i来求点积，加上的是b_h_k_i而不是b_h_k_j。原因我猜测是为了方便还是计算Axial Attention，否则就不是基于列的self-attention了。具体可以见下面图示。
            - 图示化解释
                
                ![image.png](../images/image%2047.png)
                
                - 原来的图解是错误的，其并没有忠实按照官方文档中的实现进行图示，红色框部分修改了原来途中的错误标注。
            
- 最后看看Single Attention with pair bias 如何实现。
    - 输入是token-level single representation {s_i} 和token-level pair representatoin {z_i_j} , 输出是{s_i}。
    - 主要使用{z_i_j}作为偏置，加入到{s_i}的self-attention计算中，同时{s_i}的self-attention中也增加了gating的机制。
    - 具体算法伪代码实现为：
        
        ![image.png](../images/image%2048.png)
        
        - single representation {s_i} 进行归一化计算。
        - 从{s_i} 计算特定h头的q、k、v表示：q_h_i , k_h_i, v_h_i 都经过了线性变换从c_token → c。
        - 从pair representation {z_i_j} 中计算一个偏置的值，准备应用在{s_i}的self-attention中：首先{z_i_j}进行归一化，然后进行线性变换，维度从c_z → 1 ，得到b_h_i_j。
        - 使用{s_i}计算后续用于gating的值：先对{s_i}进行线性变换c_token → c ,然后对于其中的每一个元素计算sigmoid的值，得到g_h_i。
        - 然后计算attention，实际上就是正常的self-attention，只是在计算attention score的时候，加入了来自于pair representation的偏置的值b_h_i_j，从而得到标量，A_h_i_j。
        - 然后使用v_h_j对每一个j乘以A_h_i_j，并求和，得到一个经过weighted sum的向量，然后再按元素乘g_h_i，得到特定h头上的attention结果，然后将其结果按照最后一维拼接起来，并最终通过一个线性变换，得到经过attention之后的{s_i}的结果，维度为c_token。
    - 具体的图示化解释如下：
        
        ![image.png](../images/image%2049.png)
        

