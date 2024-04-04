# 1. Download pre-trained swin transformer model (Swin-T)  
    In this project, we utilized pre-trained weights from Ze Liu et al. These weights were obtained based on their work in "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".  
    
    To comply with intellectual property regulations and express our gratitude to the original authors, we explicitly acknowledge the use of these pre-trained weights. If you are interested in more details about the model, please refer to Ze Liu et al.'s paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".  
    
    The pre-trained weights can be obtained from the following location: [https://github.com/microsoft/Swin-Transformer]: Put pretrained Swin-T into folder "pretrained_ckpt/"  
# 2. Prepare data  
    We utilized two publicly accessible datasets in this project, including the LiTs dataset and the 3D-IRCADb dataset. The LiTs dataset can be found in [https://competitions.codalab.org/competitions/17094#participate]. The 3D-IRCADb dataset can be found in [https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/].  
# 3. Environment  
    Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.  
# 4. Train/Eval/Visual  
    train.py: This file contains the code for training the model.  
    eval.py: This file contains the code for evaluating the model and calculating performance metrics.  
    visual.py: This file contains the code for generating visualizations.  

