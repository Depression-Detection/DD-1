#Overview

In the reddit-cnn-2.py file, we put the comments of Reddit users into a dataframe and "clean" the comments for our use. This dataframe contains the comment and the depression status of the commenter gathered from the survey. Next, we use Natural Language Processing and Tensorflow to train the model to detect depression. This creates the model we use.



# DD-1


### Reddit Models

#### Transformer v1.0

```diff
-31.47%
```

Accuracy: 61.64% </br>
Loss: 0.3713 </br>

#### CNN v1.0 (Best Model)

```diff
-0.65%
```

Accuracy: 92.46% </br>
Loss: 0.2267 </br>

#### CNN v1.1 (Best Model)

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
