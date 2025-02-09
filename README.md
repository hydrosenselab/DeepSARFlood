# DeepSARFlood

DeepSARFlood is a deep learning-based tool for rapid and automated flood inundation mapping using SAR data. By leveraging weak flood labels from concurrent optical imagery, our approach refines flood detection using cutting-edge Convolutional Neural Networks (CNNs) and Vision Transformer (ViT) architectures. Optimized through multitask learning and model soups, DeepSARFlood integrates a novel gain algorithm to enhance ensemble diversity and provide robust uncertainty quantification. Our results demonstrate that ViT-based and CNN-ViT hybrid models surpass traditional CNNs, achieving a state-of-the-art IoU score of 0.72 on the Sen1Floods11 dataset, setting a new benchmark for flood mapping accuracy and reliability.

This repo provides the codebase for the tool, along with the model weights.

# Methodology of DeepSARFlood

![Main fig](https://github.com/hydrosenselab/DeepSARFlood/blob/master/img.jpg)  

# Runtime example:

Click to see a video of an example runtime using DeepSARFlood GUI

[![Video](https://img.youtube.com/vi/TiVcjaTp8sE/0.jpg)](https://www.youtube.com/watch?v=TiVcjaTp8sE)

# Comparative Performance
![CleanShot 2025-02-09 at 11  43 27@2x](https://github.com/user-attachments/assets/82c9578d-b067-46b1-bf18-e429af584dc0)

# Cite As
Sharma, N.K., Saharia, M., 2025. DeepSARFlood: Rapid and Automated SAR-based flood inundation mapping using Vision Transformer-based Deep Ensembles with uncertainty estimates. Sci. Remote Sens. 100203. https://doi.org/10.1016/j.srs.2025.100203

# Major Datasets Used
- Sen1Floods11 https://github.com/cloudtostreet/Sen1Floods11

# [HydroSense Lab, IIT Delhi](https://hydrosense.iitd.ac.in/) | [Lab GitHub](https://github.com/hydrosenselab) | [Zenodo](www.Zenodo.com)
For any queries, please contact:

Nirdesh Sharma  
nirdesh@civil.iitd.ac.in  
Department of Civil Engineering      
Indian Institute of Technology Delhi   

Dr. Manabendra Saharia   
Principal Investigator, HydroSense Lab     
Assistant Professor, Dept. of Civil Engineering   
Associate Faculty, Yardi School of Artificial Intelligence  
Indian Institute of Technology Delhi   
Contact: [msaharia@iitd.ac.in](msaharia@iitd.ac.in)

<p float="left">
  <img src="https://i.imgur.com/qVGTGwT.png" width=20% height=20% />
  <img src="https://i.imgur.com/6giJLL8.png" width=20% height=20% />
</p>
