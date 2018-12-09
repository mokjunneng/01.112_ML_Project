## 01.112_ML_Project

A NLP project building a sentiment analysis system as well as a phrase chunking system for Tweets on multiple languages like EN, FR, CN and SG.

Team Member:

- Mok Jun Neng
- Rachel Gan
- You Song Shan

## Instruction

### Part2 - Emissions

Calculate emission parameters for HMM in part3.

Run following command line to start training and testing. The output file called <u>dev.p2.out</u> will be generated in [data](data) folder which contains the test set.

```shell
python part2/emission.py [train file] [dev.in file]
# for example
python part2/emission.py data/EN/train.dev data/EN/dev.in 
```

---

### Part3 - First-order HMM

Run following command line to start training and testing. The output file called <u>dev.p3.out</u> will be generated in [data](data) folder which contains the test set.

```shell
python part3/viterbi.py [train_file] [test_file]
# for example
python part3/viterbi.py data/EN/train.dev data/EN/dev.in 
```

---

### Part4 - Second-order HMM

Run following command line to start training and testing. The output file  <u>dev.p4.out</u> will be generated in [data](data) folder which contains the test set.

```shell
python part4/viterbi2.py [train_file] [test_file]
# for example
python part4/viterbi2.py data/EN/train.dev data/EN/dev.in 
```

---

### Part5 - Design Challenge

To try performance of different models, 3 different approaches had been implemented for [part5](part5) design challenge, results and explanation can be found in our final report:  

- [CRF](part5/crf-nolib.py) (Build from scratch)

  ```shell
  python part5/crf-nolib.py [train file] [dev.in file] [result filepath]
  ```

- [Perceptron](part5/structured_perceptron.py) (Build from scratch)

  ```shell
  python part5/structured_perceptron.py [train file] [dev.in file] [result filepath]
  ```
- [CRF](part5/crf-withlib.py) (Build with external ML packages)
  
  ```shell
  python part5/structured_perceptron.py [train file] [dev.in file] [result filepath]
  ```

- [MEMM](part5/MEMM.py) (Build with external ML packages)

  ```shell
  python part5/MEMM.py [train file] [dev.in file]
  ```

- [HMM](part5/HMM_turingsmoothing/viterbi.py)
  
  ```shell
  python part5/HMM_turingsmoothing/viterbi.py [train file] [dev.in file]
  ```
---

### Evaluate

To evaluate the performance using [script](evalResult.py), run following:

```shell
python evalResult.py [gold truth file] [prediction file]
```