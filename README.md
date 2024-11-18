# Multi-Trigger-Backdoor-Attacks
In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries use different types of triggers to poison the same dataset.

# Introduction
Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). While many methods have been proposed for detecting and mitigating backdoor attacks, most rely on identifying and removing the "shortcut" created by the backdoor, which links a specific source class to a target class. However, these methods can easily be bypassed by designing multiple backdoor triggers that create shortcuts everywhere, thus making detection more difficult. 

In this study, we introduce Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries use different types of triggers to poison the same dataset. We propose and investigate three types of multi-trigger attacks: \textit{parallel}, \textit{sequential}, and \textit{hybrid}. Our findings show that: 1) multiple triggers can coexist, overwrite, or cross-activate each other, and 2) MTBAs break the common shortcut assumption underlying most existing backdoor detection and removal methods, rendering them ineffective. 

Given the security risks posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to support future research on detecting and mitigating these attacks. Additionally, we discuss potential defense strategies against MTBAs.

<div align="center">
  <img src="assets/mtba_overview.png" alt="MTBA Overview" />
</div>

The figure above demonstrates the effectiveness of multi-trigger attacks at various poisoning rates (ranging from $0.2\%$ to $10\%$) under three labeling modes (All2One, All2All, and All2Random) on the CIFAR-10 dataset. The results from All2One and All2All modes show that: 1) different triggers can coexist at a $10\%$ poisoning rate with high attack success rates (ASRs), and 2) ASRs vary significantly at extremely low poisoning rates (e.g., $0.2\%$).

# Quick Start
To launch a parallel attack on ResNet-18 with 10 different triggers, run the following command:
```shell
python backdoor_mtba.py
```

# Acknowledgement
As our work is currently under review, this is an open-source project contributed by the authors for the reproduction of the main results.
