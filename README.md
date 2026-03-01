This document outlines the code execution instructions for the fully connected neural network model training process and the Bayesian optimization process covered in the article "Data-Intelligent Design of Platinum-Based High-Entropy Alloys for Efficient Methanol Oxidation Driven by a Robotic AI-Chemist".



1.Requirements

pandas 2.3.2

optuna 4.5.0

numpy 1.26.4

torch 2.6.0



2.Workflow

In the nth iteration, first copy and paste the dataset file into the same folder as the code. When running *NNmodel1\_training.py*  and *NNmodel2\_training.py* , change the input\_name in the main function to the dataset file name for the corresponding iteration (*data\_n-1.csv* ). After running the two model training codes separately, the model files *NN\_net\_1.pkl*  and *NN\_net\_2.pkl* , along with the model best parameter record files *NN\_best\_para\_1.csv*  and *NN\_best\_para\_2.csv* , will be generated in the respective folders. Then, run the code *para\_recomand.py*  to obtain the parameter recommendation file *para\_recomand\_n.csv* , where columns x1 to x5 represent the theoretical component distribution ratios, columns x6 to x10 represent the predicted actual component distribution ratios, and the score column indicates the predicted Pt unit activity under this distribution.



3.Note

Due to the introduction of the dropout mechanism during model training, some neural network nodes are randomly closed in each training session, resulting in slightly different models each time. Therefore, we provide the actual model files obtained during each training round in the model folder.




