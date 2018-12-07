## 01.112_ML_Project

A NLP project building a sentiment analysis system as well as a phrase chunking system for Tweets.

Team Member:

- Mok Jun Neng
- Rachel Gan
- You Song Shan

## Instruction

### Part2 - Emissions

Calculate emission parameters for HMM in part3.

Run following command line to start training and testing. The output file called <u>dev.p3.out</u> will be generated in [data](data) folder.

```shell
python emission.py [train file] [dev.in file]
# for example
python emission.py data/EN/train.dev data/EN/dev.in 
```

---

### Part3 - First-order HMM

Run following command line to start training and testing. The output file called <u>dev.p3.out</u> will be generated in [data](data) folder.

```shell
python viterbi.py [train_file] [test_file]
# for example
python viterbi.py data/EN/train.dev data/EN/dev.in 
```

---

### Part4 - Second-order HMM

Run following command line to start training and testing. The output file  <u>dev.p4.out</u> will be generated in [data](data) folder.

```shell
python part4/viterbi2.py [train_file] [test_file]
# for example
python part4/viterbi2.py data/EN/train.dev data/EN/dev.in 
```

---

### Part5 - Design Challenge

To try performance of different models, 3 different approaches had been implemented for [part5](part5) design challenge:  

- [CRF](part5/crf.py) (**Chosen**)

  ```shell
  python part5/MEMM.py [train file] [dev.in file]
  ```

- [Perceptron](part5/structured_perceptron.py) (with ML libraries)

  ```shell
  python part5/MEMM.py [train file] [dev.in file]
  ```

- [MEMM](part5/MEMM.py) (with ML libraries)

  ```shell
  python part5/MEMM.py [train file] [dev.in file]
  ```

---

### Evaluate

To evaluate the performance using [script](evalResult.py), run following:

```shell
python evalResult.py [dev.out] [dev.pX.out]
```