# Real-3DQA: Do 3D Large Language Models Really Understand 3D Spatial Relationships? 

<p align="center">
  <img src="https://real-3dqa.github.io/assets/logo.png" width="100">
</p>

<p align="center">
  <strong>ICLR 2026</strong>
</p>

<p align="center">
  <a href="https://xianzhengma.github.io/">Xianzheng Ma</a><sup>1,2*</sup>,
  <a href="https://taosun.io/">Tao Sun</a><sup>3*</sup>,
  <a href="https://chenusc11.github.io/">Shuai Chen</a><sup>1,2</sup>,
  <a href="https://yashbhalgat.github.io/">Yash Bhalgat</a><sup>1</sup>,
  <a href="https://jindonggu.github.io/">Jindong Gu</a><sup>5†</sup>,
  <br>
  <a href="https://angelxuanchang.github.io/">Angel X Chang</a><sup>4</sup>,
  <a href="https://ir0.github.io/">Iro Armeni</a><sup>3</sup>,
  <a href="https://scholar.google.de/citations?user=n9nXAPcAAAAJ">Iro Laina</a><sup>1</sup>,
  <a href="https://pengsongyou.github.io/">Songyou Peng</a><sup>5‡</sup>,
  <a href="https://www.robots.ox.ac.uk/~victor/">Victor Adrian Prisacariu</a><sup>2‡</sup>
</p>

<p align="center">
  <sup>1</sup>VGG, University of Oxford &nbsp;
  <sup>2</sup>AVL, University of Oxford &nbsp;
  <sup>3</sup>Stanford University
  <br>
  <sup>4</sup>Simon Fraser University &nbsp;
  <sup>5</sup>Google DeepMind
</p>

<p align="center">
  <em>* equal contribution &nbsp; † correspondence author &nbsp; ‡ equal supervision</em>
</p>

<p align="center">
  <a href="https://real-3dqa.github.io/"><img src="https://img.shields.io/badge/🌐-Project_Page-blue" alt="Project Page"></a>
  <a href="http://arxiv.org/abs/2603.23523"><img src="https://img.shields.io/badge/📝-arXiv-B31B1B" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Oliver-Ma/Real-3DQA"><img src="https://img.shields.io/badge/🤗-HuggingFace_Dataset-yellow" alt="HuggingFace"></a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=xianzhengma.Real-3DQA" alt="visitors"/>
</p>

---

This repository contains:
- 📁 Access to the **Real-3DQA dataset** (Hosted on Hugging Face).
- 📊 Two standalone **evaluation scripts** (Exact Match & Viewpoint Rotation Score).
- 🔧 The **3D-RFT (3D Reweighted Finetuning)** pseudocode.

---

## 📁 Dataset (Hugging Face)

Our dataset is hosted on Hugging Face: [Oliver-Ma/Real-3DQA](https://huggingface.co/datasets/Oliver-Ma/Real-3DQA)

You can download the JSON files directly from the repository or use the `datasets` library to load them. The dataset contains a `de-biased_testset` along with 4 rotation splits (`rotation_0`, `rotation_90`, `rotation_180`, `rotation_270`).

---

## 📊 Evaluation

We provide two standalone scripts to compute metrics. Your model's prediction file should be a JSON containing a list of dictionaries, each with at least `response_gt` and `response_pred` (and `question_id` for VRS).

### 1. Debiased EM (Exact Match)
To evaluate standard Exact Match (EM) and Relaxed Match on the debiased test set:
```bash
python evaluate_debiased_exact_match.py predictions.json --name "MyModel"
```

### 2. VRS (Viewpoint Rotation Score)
To evaluate your model's rotation consistency across all 4 viewpoints:
```bash
python evaluate_rotation_robustness.py preds0.json preds90.json preds180.json preds270.json --name "MyModel"
```

---

## 🔧 3D-RFT (3D Reweighted Finetuning) Pseudocode

We provide the implementation pseudocode for our proposed **3D-RFT** solution in [`3D-RFT.py`](3D-RFT.py). 

The key idea is to dynamically downweigh training samples where a blind model performs well, forcing the model to rely on genuine 3D spatial perception rather than linguistic shortcuts:
```python
# Core formula
loss_3drft = loss_full / loss_blind
```

---

## 📖 Citation

```bibtex
@inproceedings{ma2026real3dqa,
  title={Do 3D Large Language Models Really Understand 3D Spatial Relationships?},
  author={Xianzheng Ma and Tao Sun and Shuai Chen and Yash Bhalgat and Jindong Gu and Angel X Chang and Iro Armeni and Iro Laina and Songyou Peng and Victor Adrian Prisacariu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=3vlMiJwo8b}
}
```

---

## 📜 License

This project is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
