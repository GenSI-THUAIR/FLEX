<div align="center">
    <img src="asset/AIRlogo.png" height="80">
    <img src="asset/gensi_logo_black.png" height="80">
    <img src="asset/seed_logo.png" height="80">
</div>

---

<div align="center">

# FLEX: Inheritable Intelligence via Forward Learning from Scaling Experience

The official codebase for our paper, FLEX: Inheritable Intelligence via Forward Learning from Scaling Experience.

## Introduction
Welcome to **F**orward **L**earning from **Ex**perience (FLEX), a novel learning paradigm that shifts learning from modifying model parameters to constructing and leveraging an evolvable experience library.
By continuously expanding and refining this library, agents can progressively acquire deeper insights and knowledge, enhancing their cognitive capabilities with accumulated experiences.

We conduct extensive experiments across diverse challenging scientific domains, including Olympiad-level mathematics (AIME25), chemical retrosynthesis (USPTO50k), and protein fitness prediction (ProteinGym). FLEX demonstrates substantial and consistent improvements on these tasks, from 40\% to 63\% on AIME25 and 20\% to 30\% on USPTO50k, exhibiting great enhancement in the capacity of reasoning and knowledge leverage.

<img src="asset/front_pic.png" width="100%">

The following picture exhibits the differences between gradient-based learning and FLEX, highlighting the interaction among the actor $\pi$, updater $\mu$, and experience library $\mathcal{E}$ of FLEX.

<img src="asset/method_pic.png" width="100%">

We have also discovered **two exciting features** of FLEX:
1. **The scaling law** for the experience library: agent performance scales predictably with accumulated knowledge and revealing a path towards a collaborative experience ecosystem.
2. **Intelligence Inheritance**: Distilled experience can be transferred between agents in a plug-and-play manner, enabling instant knowledge assimilation and bypassing redundant learning
## Getting Started
We will release the code, datasets, trained experience libraries soon. Stay tuned!

## Citation

If you use FLEX in your research, please cite our paper:

```bibtex
@misc{cai2025flexcontinuousagentevolution,
      title={FLEX: Continuous Agent Evolution via Forward Learning from Experience}, 
      author={Zhicheng Cai and Xinyuan Guo and Yu Pei and JiangTao Feng and Jiangjie Chen and Ya-Qin Zhang and Wei-Ying Ma and Mingxuan Wang and Hao Zhou},
      year={2025},
      eprint={2511.06449},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.06449}, 
}
```

**Paper:** [arXiv:2511.06449](https://arxiv.org/abs/2511.06449)
