---
title: Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization
---

# 🔐 Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization

📌 **Official implementation of our CVPRW 2025 paper**  
"A PyTorch-based framework for training anonymization models with action recognition constraints."

🧠 This project introduces a **penalty-driven strategy** to balance 🔒 anonymization and 🏃‍♀️ action recognition performance, using custom 🧪 loss functions and 🛠️ two-step training.

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization?style=social)](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization/stargazers)
[![Visitors](https://visitor-badge.glitch.me/badge?page_id=Rabusi.Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization)](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization)
[GitHub Repo](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization)

---


⭐ If you find this work helpful to your research, don't forget to give a star to this repo! Thanks! 🤗

## 📄 Abstract

The rapid development of video surveillance systems for object detection, tracking, activity recognition, and anomaly detection has revolutionized our day-to-day lives while setting alarms for privacy concerns. It isn’t easy to strike a balance between visual privacy and action recognition performance in most computer vision models. Is it possible to safeguard privacy without sacrificing performance? It poses a formidable challenge, as even minor privacy enhancements can lead to substantial performance degradation. To address this challenge, we propose a privacy-preserving image anonymization technique that optimizes the anonymizer using penalties from the utility branch, ensuring improved action recognition performance while minimally affecting privacy leakage. This approach addresses the trade-off between minimizing privacy leakage and maintaining high action performance. The proposed approach is primarily designed to align with the regulatory standards of the EU AI Act and GDPR, ensuring the protection of personally identifiable information while maintaining action performance. To the best of our knowledge, we are the first to introduce a feature-based penalty scheme that exclusively controls the action features, allowing freedom to anonymize private attributes. Extensive experiments were conducted to validate the effectiveness of the proposed method. The results demonstrate that applying a penalty to anonymizer from utility branch enhances action performance while maintaining nearly consistent privacy leakage across different penalty settings.           

## 🧩 Proposed Penalty-Driven Framework

![Architecture](images/architecture.png)

## 🖼️ Anonymized Images

### Brush Hair
![Brush Hair](images/brushhair.png)

## 📊 Results
![](images/tab1.png)
![](images/tab2.png)

## 🎥 Raw vs Anonymized Video Comparison

<div style="display: flex; gap: 20px; flex-wrap: wrap;">
  <div>
    <h4>🔍 Raw Video</h4>
    <video width="320" height="240" controls>
      <source src="images/1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

  <div>
    <h4>🕶️ Anonymized Video</h4>
    <video width="320" height="240" controls>
      <source src="images/2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

  <div>
    <h4>🔍 Raw Video</h4>
    <video width="320" height="240" controls>
      <source src="images/3.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

  <div>
    <h4>🕶️ Anonymized Video</h4>
    <video width="320" height="240" controls>
      <source src="images/4.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

## 📬 Contact

For questions or feedback, feel free to reach out at 📧 [naas@create.aau.dk](mailto:naas@create.aau.dk)
