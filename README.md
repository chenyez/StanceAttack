# StanceAttack
This repository includes codes/data of our ACL'26 Findings paper: "StanceAttack: Adversarial Attack for Stance Detection"

Please cite our paper at:

@inproceedings{zhao-caragea-2026-stanceattack,
    title = "{S}tance{A}ttack: Adversarial Attack for Stance Detection",
    author = "Zhao, Chenye  and
      Caragea, Cornelia",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.2034/",
    doi = "10.18653/v1/2026.findings-acl.2034",
    pages = "40954--40971",
    ISBN = "979-8-89176-395-1",
    abstract = "Stance detection aims to ascertain whether an author{'}s text is in favor, against, or neutral toward specific targets like public policies or social issues. While pretrained language models (PLMs) have greatly enhanced stance detection, they remain vulnerable to adversarial attacks{---}manipulations that maintain textual semantics but lead to incorrect predictions. Such vulnerabilities remain underexplored for stance detection. In this study, we introduce StanceAttack, an innovative adversarial attack method leveraging ChatGPT to create adversarial examples that can mislead well-trained stance detection models. We conduct experiments to evaluate our attack model by attacking state-of-the-art PLMs on two benchmark datasets. Results demonstrate that StanceAttack outperforms traditional adversarial methods with higher success rates and fewer retries. Human evaluations confirm that our adversarial examples preserve the original semantic meanings and naturalness. We share our code and data in https://github.com/chenyez/StanceAttack."
}
