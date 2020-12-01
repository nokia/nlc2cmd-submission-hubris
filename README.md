This is the code for the submission of Jaron Maene (team name `Hubris`) to the NeurIPS [nlc2cmd competition](http://nlc2cmd.us-east.mybluemix.net/). The code/model was partially developed during a summer internship at Nokia Bell Labs and was further improved for the competition. You can find the nl2cmd leaderboard [here](https://eval.ai/web/challenges/challenge-page/674/leaderboard/1831). `Hubris` finished in second place. 


# Disclaimer 

This repository is not an official Nokia product. It contains research code that was partially developed during an internship at Nokia Bell Labs. It is solely made for research purposes.

# Installation

Clone this repo and install the python dependencies with:

```
pip install -r requirements.txt
```

If you just want to run inference, you only really need to install `pytorch` and `transformers`.

If you want to train the models from scratch, you'll also need to download the datasets, as these are not included in the repo. You might want to check the data-preprocessing notebook, which has the sources of all data, plus the code to preprocess it. 

# Usage

The `eval.py` is a self-contained script that contains all code you need to perform inference on the model(s). You can use it with:
```
python eval.py <COMMANDS_FILE>
```
Where `<COMMANDS_FILE>` is a text file that contains 1 natural language command per line. As an example, you could use `data/clai/dev_nl.txt`.
Note that this does inference on cpu (which was required in the nlc2cmd competition). So expect this to be (very) slow.

# Training

To run the experiments defined in `src/tune.py`:
```
python train.py
```
You can change the hyperparameters in the `src/tune.py`. Using the current ones, you'll train the same gpt2-large model that was used in the competition. Refer to `src/config.py` for an explanation on the meaning of the different hyperparameters. In the notebooks, you can find some graphs on these used for tuning.

# Web app
The code comes with a basic flask app to demo the model. You can start it with:
```
flask run
```
