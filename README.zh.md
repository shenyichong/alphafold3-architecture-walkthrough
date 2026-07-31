# 超详细 AlphaFold 3 算法解析

by Yichong Shen

> **📧 联系方式**: 如果您对本分析有任何问题、讨论或合作意向，欢迎通过以下邮箱联系：shenyichong2011@gmail.com
> 
> **⭐ 如果您觉得这个分析对您有帮助，请考虑为本仓库点个star！您的支持能帮助更多人发现这个资源。**

> 本技术深度解析的灵感来源于 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)。特别感谢 Elana Pearl 提供的可视化资源。

**📖 阅读语言：** [English](README.md) | [中文](README.zh.md)

**💬 讨论：** 加入我们的[Reddit讨论](https://www.reddit.com/r/bioinformatics/comments/1l7xcp3/alphafold_3_demystified_i_wrote_a_technical/)，与生物信息学社区一起讨论这个话题。

---

## 📋 目录

- [Input Preparation](docs/Input%20Preparation.zh.md)
  - [**MSA和Templates是如何来的？**](docs/Input%20Preparation.zh.md#msa和templates是如何来的)
    - [为什么需要MSA？](docs/Input%20Preparation.zh.md#为什么需要msa)
    - [为什么需要Templates？](docs/Input%20Preparation.zh.md#为什么需要templates)
    - [如何获取MSA？](docs/Input%20Preparation.zh.md#如何获取msa)
    - [如何获取Templates？](docs/Input%20Preparation.zh.md#如何获取templates)
    - [如何表征Templates？](docs/Input%20Preparation.zh.md#如何表征templates)
  - [**如何构建Atom-level的表征？**](docs/Input%20Preparation.zh.md#如何构建atom-level的表征)
    - [构建p和q](docs/Input%20Preparation.zh.md#构建p和q)
    - [更新q（Atom Transformer）](docs/Input%20Preparation.zh.md#更新qatom-transformer)
  - [**如何构建Token-level的表征？**](docs/Input%20Preparation.zh.md#如何构建token-level的表征)
    - [Token级单序列表征](docs/Input%20Preparation.zh.md#token级单序列表征)
    - [Token级对表征](docs/Input%20Preparation.zh.md#token级对表征)
- [Representation Learning](docs/Representation%20Learning.zh.md)
  - [Template Module](docs/Representation%20Learning.zh.md#template-module)
  - [MSA Module](docs/Representation%20Learning.zh.md#msa-module)
  - [Pairformer Module](docs/Representation%20Learning.zh.md#pairformer-module)
- [Structure Prediction](docs/Structure%20Prediction.zh.md)
  - [Diffusion的基本概念](docs/Structure%20Prediction.zh.md#diffusion的基本概念)
  - [Structure Prediction 详解](docs/Structure%20Prediction.zh.md#structure-prediction-详解)
    - [Sample Diffusion部分详解（推理过程）](docs/Structure%20Prediction.zh.md#sample-diffusion部分详解推理过程)
    - [Diffusion Module部分详解（推理过程）](docs/Structure%20Prediction.zh.md#diffusion-module部分详解推理过程)
- [Loss Function](docs/Loss%20Function.zh.md)
    - [$L_{distogram}$](docs/Loss%20Function.zh.md#l_distogram)
    - [$L_{diffusion}$](docs/Loss%20Function.zh.md#l_diffusion)
    - [$L_{confidence}$](docs/Loss%20Function.zh.md#l_confidence)


---

# 📬 联系与合作

**作者**: Yichong Shen  
**邮箱**: shenyichong2011@gmail.com  
**GitHub**: [@shenyichong](https://github.com/shenyichong)

如果您对这份AlphaFold 3分析有任何问题、讨论或潜在合作意向，请随时通过邮箱联系或在本仓库中创建issue。

# 🌟 支持这项工作

如果这份全面的分析帮助您更好地理解了AlphaFold 3，请考虑：
- ⭐ **为本仓库点星** 帮助其他人发现它
- 🔄 **分享** 给您的同事和网络
- 💭 **创建issues** 提出问题或建议  
- 🤝 **贡献** 通过pull request提供改进或修正

---