# PropTest: Automatic Property Testing for Improved Visual Programming

This is the code for the paper [PropTest: Automatic Property Testing for Improved Visual Programming](https://jaywonkoo17.github.io/PropTest/) 

### [Paper (EMNLP 2024 Findings)](https://arxiv.org/pdf/2403.16921) | [Project Page](https://jaywonkoo17.github.io/PropTest/)

# Environmnet

---
Clone recursively:
```bash
git clone --recurse-submodules https://github.com/uvavision/PropTest.git
```

After cloning:
```bash
cd PropTest
export PATH=/usr/local/cuda/bin:$PATH
bash setup.sh  # This may take a while. Make sure the vipergpt environment is active
cd GLIP
python setup.py clean --all build develop --user
cd ..
echo YOUR_OPENAI_API_KEY_HERE > api.key
```
This code was built on top of [ViperGPT](https://github.com/cvlab-columbia/viper). We follow the same installation steps as ViperGPT. For detailed installation, please refer to the [ViperGPT repository](https://github.com/cvlab-columbia/viper).

You need to download two pretrained models and store it in ```./pretrained_models```. 
You can use ```download_models.sh``` to download the models.

# Running the Code

---

The code can be run using the following command:
    
```
CONFIG_NAMES=your_config_name python main_batch.py
```
```CONFIG_NAMES``` is an environment variable that specifies the configuration files to use.

# Citation

---
Please cite our paper if you find our method or code useful:
```
@article{koo2024proptest,
      title={PropTest: Automatic Property Testing for Improved Visual Programming}, 
      author={Jaywon Koo and Ziyan Yang and Paola Cascante-Bonilla and Baishakhi Ray and Vicente Ordonez},
      journal={arXiv preprint arXiv:2403.16921},
      year={2024}
}
```