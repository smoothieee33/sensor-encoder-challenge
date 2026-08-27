# The Lovely... Sensor Encoder Challenge!!
welp i lowk forgot github existed so heres a big ol push i hope yall like me anyway ik this is bad practice TwT

This is the sensor encoder challenge for HCL's ML internship. The project is almost compare and contrasty, you take data from UCI's Human Activity Recognition Using Smartphones datasets and process it two ways: a small frozen LM, and an ordinary sensor classifier, (basically what you would think of when you think data classification). Lastly, you take the macro F1 for the classifier, LM, and the LM again but you shuffle the embeddings to make sure its actually using the embeddings/sensor data given. (the F1 of the second and third should match if that was the case). 

Heres how to set ts up and reproduce it!! In your terminal:

    git clone https://github.com/smoothieee33/sensor-encoder-challenge
    
    cd sensor-encoder-challenge
    
    python3 -m venv venv
    
    pip install -r requirements.txt
    
    curl -L -o har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    
    unzip har.zip -d har_raw
    
*Known Issue!* when downloading HuggingFaceTB/SmolLM2-360M-Instruct for the context embedding model, disable "xet."   There's a problem with their backend, causing downloads to hang indefinetly. Use:

    export HF_HUB_DISABLE_XET=1

As this forces a plain http download, skipping the xet. Set this once every terminal session or add to ~/.zshrc or ~/.barshrc

Next!! Run these files in order! I used python version 3.9.6

- python3 train.py // trains the direct classifier and makes a best_model.pt file. Trains for 50 epochs. Seed: 67 (get it?)
- python3 evaluate_direct.py // uses best_model.pt to get result #1!!
- python3 train_context.py // trains the context model and makes a best_context_model.pt. Trains for 10 epochs. Seed: 67 (won't let it die)
- python3 evaluate_context.py // uses best_context_model.pt to get result #2!!
- python3 shuffle_check.py// uses best_context_model.pt again to get result #3!! Seed: 67 (let me be a kid ok)

RESULTS!!
| Condition | Macro-F1 |
|---|---|
| Direct sensor classifier | 0.8943 |
| Context-embedding model | 0.8935 |
| Context model, shuffled embeddings | 0.1644 |
