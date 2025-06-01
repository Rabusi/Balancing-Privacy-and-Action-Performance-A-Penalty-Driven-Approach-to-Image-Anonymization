---
title: "Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization"
---

# Balancing Privacy and Action Performance  
### A Penalty-Driven Approach to Image Anonymization  
**Nazia Aslam**<sup>1,2</sup>, **[Kamal Nasrollahi](https://vbn.aau.dk/en/persons/117162)**<sup>1,2,3</sup>  
<sup>1</sup>Visual Analysis and Perception Lab, Aalborg University, Denmark  
<sup>2</sup>Pioneer Centre for AI, Denmark  
<sup>3</sup>Milestone Systems, Denmark  

---

[![Code](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization) 
[![arXiv](https://img.shields.io/badge/arXiv-2504.14301-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2504.14301)

---

## 🔍 Overview

This project introduces a **penalty-driven strategy** to balance image **anonymization** 🔒 and **action recognition** 🏃‍♀️ using:

- Custom loss functions
- Two-step training
- Penalty signals from the utility branch

Designed with GDPR and EU AI Act compliance in mind, our method anonymizes private features while preserving task-relevant cues.

---

## 📄 Abstract

Video surveillance has revolutionized activity recognition but raised serious privacy concerns. We address the challenge of safeguarding personal identity **without sacrificing model performance**. Our **feature-based penalty scheme**:

- Minimizes **privacy leakage**
- Maximizes **action recognition accuracy**
- Provides control over private vs. task-relevant features

> We are the first to propose a privacy-utility balancing method that applies **penalties directly on feature space**, not on output labels.

---

## 🧠 Proposed Framework

<img src="images/architecture.png" alt="Model Architecture" width="100%"/>

---

## 🖼️ Anonymized Examples

**Example: Brush Hair**

<img src="images/brushhair.png" alt="Anonymized Brush Hair" width="70%"/>

---

## 📊 Results

<img src="images/tab1.png" alt="Quantitative Results Table 1" width="85%"/>  
<img src="images/tab2.png" alt="Quantitative Results Table 2" width="85%"/>

---

## 🎥 Raw vs Anonymized Video Comparison

| Raw Video | Anonymized Video |
|-----------|------------------|
| <video width="320" height="240" controls><source src="images/1.mp4" type="video/mp4"></video> | <video width="320" height="240" controls><source src="images/2.mp4" type="video/mp4"></video> |
| <video width="320" height="240" controls><source src="images/3.mp4" type="video/mp4"></video> | <video width="320" height="240" controls><source src="images/4.mp4" type="video/mp4"></video> |

---

## 📚 Citation

If you find our work useful, please cite it as:

```bibtex
@article{aslam2024balancing,
  title={Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization},
  author={Aslam, Nazia and Nasrollahi, Kamal},
  journal={arXiv preprint arXiv:2504.14301},
  year={2024}
}
```

---

## 📬 Contact

Have questions or want to collaborate?  
📧 [naas@create.aau.dk](mailto:naas@create.aau.dk)
