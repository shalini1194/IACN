# IACN
This is code for paper titled- IACN: Influence-aware and Attention-basedCo-evolutionary Network for Recommendation
## Code setup 

To initialize the directories needed to store data and outputs, use the following command. This will create data/, saved_models/, and results/ directories.
```
./initialize.sh
```
## Running the code
To train the IACN model using the data/<network>.csv dataset,, run the following command. This will save a model for every epoch in the saved_models/<network>/ directory
```
python IACN.py --network <network> --epochs 50
```

## Evaluate the model
To evaluate the performance of the model, use the following command. It tests the model performance saved at $ep$th epoch.
  ```
  python IACN_test.py --epoch ep
  ```
