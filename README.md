# Concurrency Data Race Detector
Simple script to generate steps for detecting data races in multi threaded programs

## Running instructions
Run ```pip install -r requirements.txt```, followed by ```python3 datarace.py FILE_NAME```. The file must have the following structure:
```
NO_THREADS
NAME_OF_LOCK_1 NAME_OF_LOCK_2 ...
NAME_OF_VAR_1 NAME_OF_VAR_2 ...
COMMAND_1
COMMAND_2
...
```
See ```example.txt``` for an example.
