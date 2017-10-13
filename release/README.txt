README.txt

:Author: zq
:Email: theodoruszq@gmail.com
:Date: 2017-10-12 21:15


1. First you need to download ·有雾.zip· and ·无雾.zip· datasets and unzip them all in `train/fog` and `train/notfog` directories

[1.1 You NEED to make sure the filenames of datasets are legal, just go deep and check them]

2. Create `train/fog`, `train/notfog`, `valid/fog`, `valid/notfog` `test/fog`, `test/notfog` directories in PROJECT ROOT folder
    OR you can run `init_dir.sh` to complete this

    2.1 Make filenames legal
        Run `fnstd train/fog` and `fnstd train/notfog`
    2.2 Make training and validation and test data
        Run `python3 move2test.py` and `python3 move2valid.py`

    NOTE: These scipt are all in `utils` folder, you may need to copy them to PROJECT ROOT directory

3. Run `python3 `

4. Run `python3 resnet18_avg.py` to get our classifier

5. Run `python3 eval_resnet18_avg.py` to check the accuracy of our classifier


NOTE: This model is affined to ResNet18, which is an amazing model in Computer Vision area.





