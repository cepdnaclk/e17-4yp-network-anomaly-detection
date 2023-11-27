---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e17-4yp-network-anomaly-detection
title: Structure Learning in Graph Attention Network based Network Anomaly Detection
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Structure Learning in Graph Attention Network based Network Anomaly Detection

#### Team

- E/17/044, Coralage D.T.S, [email](e17044@eng.pdn.ac.lk)
- E/17/251, Perera S.S , [email](e17251@eng.pdn.ac.lk)
- E/17/252,  Perera U.A.K.K, [email](mailto:e17252@eng.pdn.ac.lk)

#### Supervisors

- Prof. Roshan Ragel, [email](roshanr@eng.pdn.ac.lk)
- Dr. Suneth Namal, [email](sunethn@pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Related works](#related-works)
4. [Methodology](#methodology)
5. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)
8. [Links](#links)



## Abstract
Network anomaly detection is a critical aspect of ensuring the security and stability of differnt kinds of networks.
      Graph Attention Networks (GATs) have shown promise in modeling complex relationships within graph-structured data, making them well-suited for capturing intricate dependencies in network connections. This research introduces an innovative approach to enhance network anomaly detection through the incorporation of structure learning within GATs. The proposed method leverages the inherent ability of GATs to discern important nodes and edges in a network graph while introducing a mechanism for adaptive structure learning based on causality. By dynamically adjusting attention weights during the training process, our model autonomously identifies and prioritizes key structural nodes, facilitating a more robust representation of network behavior. This adaptability aims to improve the model's ability to discriminate between normal and anomalous network patterns while reducing the false detection rate.
    

## Introduction

Anomaly detection is the process of identifying deviations from what is considered normal behavior, a concept applicable across various disciplines. Over the years, researchers have employed diverse methods to detect anomalies in multivariate time series data. Modern  networks have increased in scale and complexity. This has led to the need for a proper self-operating network monitoring and detection mechanism to ensure the reliability of the network. Early approaches to network anomaly detection focused on rule-based systems that rely on predefined thresholds defined by network operators with expertise. However, these methods are limited in their ability to adapt to more diverse and complex networks. With the voyage of machine learning and data mining techniques, researchers began exploring statistical analysis and data-driven algorithms to identify anomalies. In recent years, network anomaly detection shifted towards more sophisticated approaches, including deep learning, neural networks, and graph-based approaches to model network anomalies in complex networks.
Deep learning models have gained popularity, particularly in complex networks with high-dimensional features. Surveys suggest that neural networks, augmented with attention mechanisms to capture dependencies in time series data, are more effective. Additionally, graph-based deep learning approaches excel in capturing potential dependencies between nodes, considering contextual information and improving prediction accuracy compared to methods overlooking the inherent graph structure of computer networks or treating all data uniformly.
Nevertheless, graph neural networks necessitate the input of the graph structure, posing a challenge in capturing intricate inter-sensor relationships and identifying anomalies deviating from these relationships. In our study, we scrutinize existing structure-learning approaches and propose a methodology for anomaly detection employing graph neural networks and causal-guided structure learning while reducing the false detections in the anomaly detection process.


## Related works
Historically, network anomaly detection research has predominantly concentrated on intrusion detection, leaving performance monitoring with limited consideration. This incongruity prompted our investigation into the fundamental differences between these two domains. The demarcation between them can be unclear, given the frequent intersection of security and reliability contexts. Anomalies related to performance usually result from malfunctions and misconfigurations, whereas security-related anomalies arise from deliberate malicious activities seeking to interfere with the regular functioning of a network.
An uncomplicated criterion for classifying network anomaly detection involves the method employed for anomaly detection. Over several decades of research, diverse methodologies have emerged, including statistical models, mathematical algorithms, as well as machine learning and deep learning approaches. Research referenced as [20] has employed statistical models based on mathematical algorithms like EWMA and Holt-Winters. The Work discussed in [21] discusses the Gaussian Mixture model to estimate parameters based on EM algorithms. Work in [22] is based on Bayesian networks. But researchers have drifted away from statistical approaches due to the limitations in feature representations and making assumptions. They have focused on machine learning based methodologies for anomaly detection. There are many researches for anomaly detection based on various machine learning techniques such as classfication, distance-based, clustering and etc. [23], [24], [25] and [27] are some of research work based on above mentioned techinques with algorithms Support Vector Machine, K-means clustering, random forests and K-Nearest Neighbors respectively.



## Methodology

## Experiment Setup and Implementation



## Results and Analysis


## Conclusion



## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e17-4yp-network-anomaly-detection)
- [Project Page](https://cepdnaclk.github.io/e17-4yp-network-anomaly-detection/ )
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"