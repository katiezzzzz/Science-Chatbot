# Science Robot

## This is a robot that can answer scientific questions
The main files of the robot is inside openfabric-test folder.

### Training
Datasets: wikipedia pages for feature extraction and SciQ for model training
- install relavent libraries
- check [SciQ_dataset](https://allenai.org/data/sciq) is available (stored inside `openfabric-test/`)
- generate wikipedia data using `openfabric-test/code_files/data_gen.py`
- train the model with specified parameters by executing `openfabric-test/run_nn.py`

### Testing and generating robot response
- first set proj_name to the path of the trained model
- locally by running the `openfabric-test/start.sh` 
- on in a docker container using `openfabric-test/Dockerfile` 

Robot can also be tested in command-line by running `openfabric-test/tdif_chatbot.py` 