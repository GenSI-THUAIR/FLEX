

# FLEX: Continuous Agent Evolution via Forward Learning from Experience

[![arXiv](https://img.shields.io/badge/arXiv-2511.06449-b31b1b.svg)](https://arxiv.org/abs/2511.06449)

</div>

The official codebase for our paper, FLEX: Continuous Agent Evolution via Forward Learning from Experience.




## ProteinGYM Dataset

Pro-pocessed proteingym targets [here](https://zenodo.org/records/20592649). 



## Environment

```bash
pip install pandas biomni langchain_openai tabulate scipy joblib
```

env configuration file saved in `env.yml`


## Quick-Start

### STEP 1: Experience collection

```bash
python run_self_evolve.py --setting {zero-shot / few-shot} --input_dir {processed_data_dir} --base_url xxxx --api_key xxxx
```

this will create a sub-dir under /logs .

### STEP 2: evaluation with experience library


```bash
python forward_learning/{setting}gather_evolve_result_long.py --input_dir {log_dir} --data_dir {processed_data_dir}
```




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
