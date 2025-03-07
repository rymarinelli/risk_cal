# Adversarial Prompt Filtering Based on Societal Risk 

This repo is for the "Adversarial Prompt Filtering Based on Societal Risk" paper. To install dependencies use ```pip install -r requirements.txt``` 

The research was mostly conducted on Google Colab. The ```notebooks``` directory contains the notebooks used in the experiments. This repo is meant to be more productionized to 
more easily share and distribute. 

The ```--use_categorical``` arg will calculate harm based on cateogories. It will produce the graphs of interest

```{bash}
python /content/risk_cal-main/risk_cal-main/src/main.py --use_categorical --num_epochs=10
```

Without passing the arg, it will default to the global centroid. 

```{bash}
python /content/risk_cal-main/risk_cal-main/src/main.py --num_epochs=100
```

The ```data``` directory has snapshots of some of the news summaries. It might be interesting to review to see what was being calibrated on. 

