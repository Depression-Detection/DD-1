# Overview

In the reddit-cnn-2.py file, we put the comments of Reddit users into a dataframe and "cleaned" the comments for our use. This dataframe contains the comment text and the depression status of the commenter which was gathered based off of the DSM-5 criteria from our survey. With this data, we use Tensorflow software and Natural Language Processing techniques to train a Convulational Neural Network to detect depression. This model is stored in the reddit-cnn-2.zip file.



# DD-1


### Reddit Models

#### Transformer v1.0

```diff
-31.47%
```

Accuracy: 61.64% </br>
Loss: 0.3713 </br>

#### CNN v2.0 (Best Model)

```diff
-0.65%
```

Accuracy: 92.46% </br>
Loss: 0.2267 </br>

#### CNN v2.0 (Best Model)

```diff
+0.00%
```

Accuracy: 93.11% </br>
Loss: 0.2510 </br>

### Emotion Models

#### Sadness CNN v1.0

```diff
+0.00%
```

Accuracy: 95.50%
Loss: 0.1547
