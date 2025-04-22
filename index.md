---
title: "Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization"
--- 

[Nazia Aslam](https://github.com/Rabusi)<sup>1,2</sup>, [Kamal Nasrollahi](https://vbn.aau.dk/en/persons/117162)<sup>1,2,3</sup>  
<br>
<sup>1</sup> Visual Analysis and Perception Lab, Aalborg University, Denmark, <sup>2</sup> Pioneer Centre for AI, Denmark, <sup>3</sup> Milestone Systems, Denmark


🧠 This project introduces a **penalty-driven strategy** to balance 🔒 anonymization and 🏃‍♀️ action recognition performance, using custom 🧪 loss functions and 🛠️ two-step training.

---

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=visits&edge_flat=false)](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization) [![Code](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Rabusi/Balancing-Privacy-and-Action-Performance-A-Penalty-Driven-Approach-to-Image-Anonymization) [![arXiv](https://img.shields.io/badge/arXiv-2504.14301-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2504.14301)

---


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
