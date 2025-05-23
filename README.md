# This is the source code of IJCNN 2024 paper TPN: Transferable Proto-Learning Network towards Few-shot Document-Level Relation Extraction

## Dependencies

- torch
- transformers 
- tqdm
- wandb

## Reproducing Results


### FREDo_single_1doc

```
sh fredo_single_1doc.sh
```


```
Expected output:
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support
P279              0.47        2.21        0.77    136
P17              30.52        5.81        9.76    59312
P495             12.30       15.73       13.80    1081
P463              7.85        6.90        7.34    493
P674              4.65       16.67        7.27    348
P361              3.28        4.70        3.87    893
P364              1.42       40.00        2.75    65
P582              1.93       33.66        3.65    101
P118             15.57       48.62       23.59    253
P1001             7.31       17.33       10.28    375
P102             15.50       23.36       18.64    428
P25               5.09       40.96        9.05    83
P194              8.91       24.32       13.04    222
P35              12.91       22.97       16.53    222
P140             12.99       12.23       12.60    458
P39               2.55       18.42        4.49    38
P272              5.31       45.62        9.51    160
P3373             6.40        6.82        6.61    645
-                    -           -           -    -
macro             8.61       21.46        9.64
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support
USED-FOR          5.52       27.62        9.20    42142
PART-OF           0.72        6.33        1.29    2529
FEATURE-OF        0.74       10.71        1.38    2465
CONJUNCTION        4.39       18.21        7.07    6414
EVALUATE-FOR        1.62       17.14        2.96    4223
HYPONYM-OF        3.39       12.27        5.31    4482
COMPARE           2.59       12.03        4.26    2328
-                    -           -           -    -
macro             2.71       14.90        4.50
```

### FREDo_single_3doc

```
sh fredo_single_3doc.sh
```


```
Expected output:
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support
P361              4.46        5.52        4.93    887
P674              4.20       25.78        7.23    128
P17              31.91        7.68       12.38    33229
P495             11.57       17.18       13.83    850
P140             12.24       10.53       11.32    171
P39               0.49       50.00        0.97    2
P463              7.62        8.81        8.17    193
P364              2.98       67.95        5.71    78
P272              3.82       61.82        7.20    55
P3373             6.60       11.20        8.31    241
P25               2.47       73.68        4.79    19
P194              7.65       25.86       11.81    116
P118              9.63       61.70       16.67    94
P582              1.55       50.00        3.02    36
P1001             6.27       29.94       10.37    167
P279              0.93       13.56        1.75    59
P102             11.00       19.71       14.12    279
P35               9.84       10.81       10.30    111
-                    -           -           -    -
macro             7.51       30.65        8.49
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support
USED-FOR          5.65       32.62        9.63    14548
CONJUNCTION        4.68       21.34        7.67    3332
PART-OF           0.75        7.79        1.37    1335
EVALUATE-FOR        1.88       19.52        3.42    2382
FEATURE-OF        0.74       15.47        1.41    1235
HYPONYM-OF        3.59       12.03        5.53    2311
COMPARE           3.15       13.77        5.12    1162
-                    -           -           -    -
macro             2.92       17.50        4.88
```

### ReFREDo_single_1doc

```
sh refredo_single_1doc.sh
```


```
Expected output:
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support
P17              56.38       10.98       18.38    182432
P1001            20.38       13.48       16.23    5814
P35              14.47       25.97       18.59    847
P361             13.11        2.52        4.23    12974
P463             17.18        5.89        8.78    7363
P3373            11.71       20.57       14.92    1332
P25               7.49       30.85       12.05    282
P582              8.69       42.51       14.43    327
P495             32.47       23.79       27.46    3851
P102             12.44       18.27       14.80    602
P118             36.34       35.79       36.06    665
P279              7.77        4.34        5.57    553
P674             15.39       14.38       14.87    1175
P364              7.88       44.41       13.39    358
P140             14.33        5.14        7.57    953
P272             11.11       51.86       18.30    430
P39               3.52        8.33        4.95    84
P194             19.27       11.98       14.78    701
-                    -           -           -    -
macro            17.22       20.62       14.74
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support
USED-FOR          6.88       18.99       10.10    42142
PART-OF           1.20        4.67        1.91    2529
FEATURE-OF        0.85        3.89        1.40    2465
CONJUNCTION        5.35        9.29        6.79    6414
EVALUATE-FOR        2.08        9.33        3.40    4223
HYPONYM-OF        4.16        9.46        5.78    4482
COMPARE           4.55       15.42        7.03    2328
-                    -           -           -    -
macro             3.58       10.15        5.20   
```

### ReFREDo_single_3doc

```
sh refredo_single_3doc.sh
```

```
Expected output:
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support
P1001            19.84       16.57       18.06    2927
P35              16.22       25.30       19.77    502
P17              56.14       16.01       24.92    82184
P361             10.65        2.50        4.05    8357
P463             14.46        7.11        9.53    4333
P364              4.65       59.02        8.62    122
P25               8.10       49.28       13.91    138
P3373            14.74       20.99       17.32    986
P582              7.95       41.03       13.32    156
P279              5.58       10.62        7.32    226
P102             10.70       22.50       14.50    280
P495             25.05       21.80       23.31    1913
P140             29.11        8.72       13.42    791
P118             28.78       55.32       37.86    282
P194             20.38       20.71       20.54    367
P674             17.45       10.45       13.07    708
P39               3.47       14.29        5.58    49
P272             10.90       42.68       17.36    239
-                    -           -           -    -
macro            16.90       24.72       15.69
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support
USED-FOR          7.13       24.22       11.02    14548
PART-OF           0.86        4.12        1.43    1335
HYPONYM-OF        4.34       10.82        6.20    2311
FEATURE-OF        1.01        6.72        1.76    1235
EVALUATE-FOR        2.60       12.89        4.32    2382
CONJUNCTION        6.29       12.27        8.32    3332
COMPARE           5.30       18.85        8.27    1162
macro             3.93       12.84        5.90
```

### ReFREDo_hard_1doc

```
sh refredo_hard_1doc.sh
```

```
Expected output:
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support
P17              52.08       49.87       50.96    182432
P361             13.50        3.02        4.94    12974
P463             15.21        4.89        7.40    7363
P1001             9.57       24.30       13.73    5814
P35              19.23       25.38       21.88    847
P3373            20.68       38.59       26.93    1332
P25               9.56       39.72       15.41    282
P582             12.97       43.12       19.94    327
P495             23.49       45.70       31.04    3851
P364              7.77       44.13       13.22    358
P102             11.02       13.95       12.32    602
P39               2.80        8.33        4.19    84
P118             33.61       24.66       28.45    665
P279              7.03        4.70        5.63    553
P674             20.56       16.85       18.52    1175
P272             11.55       49.07       18.70    430
P140             16.18        2.94        4.97    953
P194             39.32        6.56       11.25    701
-                    -           -           -    -
macro            18.12       24.77       17.19
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support
USED-FOR          6.82       15.17        9.41    42142
PART-OF           1.00        3.32        1.54    2529
FEATURE-OF        0.59        2.47        0.96    2465
CONJUNCTION        3.49        9.98        5.17    6414
EVALUATE-FOR        2.33        6.75        3.47    4223
HYPONYM-OF        3.55        6.67        4.63    4482
COMPARE           3.32       13.62        5.34    2328
-                    -           -           -    -
macro             3.02        8.28        4.36
```

### ReFREDo_hard_3doc

```
sh refredo_hard_3doc.sh
```

```
Expected output:
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support
P1001             9.33       28.12       14.01    2927
P17              49.80       62.92       55.59    82184
P361             12.19        3.18        5.05    8357
P463             12.37        6.25        8.31    4333
P495             22.40       47.88       30.52    1913
P35              21.06       22.11       21.57    502
P25              10.35       52.90       17.32    138
P3373            24.09       38.84       29.74    986
P279              4.80       10.18        6.52    226
P582             12.77       41.67       19.55    156
P140             15.79        3.41        5.61    791
P364              5.50       71.31       10.21    122
P102              8.84       13.93       10.82    280
P118             24.90       42.20       31.32    282
P39               2.00       10.20        3.34    49
P674             23.33       12.85       16.58    708
P194             26.35       10.63       15.15    367
P272             10.86       41.42       17.20    239
-                    -           -           -    -
macro            16.48       28.89       17.69
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support
USED-FOR          7.24       19.89       10.61    14548
HYPONYM-OF        3.70        7.66        4.99    2311
FEATURE-OF        0.79        4.94        1.36    1235
PART-OF           1.06        4.94        1.75    1335
CONJUNCTION        4.01       13.93        6.23    3332
COMPARE           3.50       16.27        5.76    1162
EVALUATE-FOR        2.83        8.82        4.29    2382
-                    -           -           -    -
macro             3.30       10.92        5.00
```


