# vit-position-attack

**select 14 patches to attack**

FGSM: Select the 14 patches with the largest gradient for FGSM attack

PGD: Select the 14 patches with the largest gradient for PGD attack

Fixed-: 14 patches were manually selected according to the maximum gradient distribution

Random-: 14 patches were randomly selected

ACC

|method  |FGSM|FixedFGSM|RandomFGSM|PGD|FixedPGD|RandomPGD|
|--------|----|---------|----------|---|--------|---------|
|cifar100|67.72|72.02   |77.70     |22.14|**7.16**|61.43  |

**Test PGD on different datasets**

select 14 patches

|datasets\method|PGD|FixedPGD|
|---------------|---|--------|
|cifar100       |22.14|**7.16**|
|food101        |24.81|**16.97**|
|beans          |34.17|**16.67**|


**Test PGD attacks with different patches**

on cifar100

|patches|FixedPGD|
|-------|--------|
|14|**7.16**|
|8|**11.86**|
|4|**27.47**|

