# Training Scripts

* [**Introduction**](#introduction)
* [**Single File Processor**](#single_processor)
* [**Annotation Based File Processor**](#annotation_processor)
  * [**Stream Object Annotation Format**](#stream_annotation_format)
* [**Trainer**](#trainer)

## <a name="introduction">Introduction

Training scripts are used to process the raw data and convert it into trainable format. After that, training of subject model is initiated and after training trained resources are saved into some persistent storage.

## <a name="single_processor">Single File Processor

Single file processor is used to convert video file into training data, against given label. This [script][sfp] take following arguments as input:

```bash
usage: single_processor.py [-h] -l LABEL [-v VIDEO] [-t TARGET] [-sk SKIP]
                           [-q QUALITY] [-tw TARGET_WIDTH] [-th TARGET_HEIGHT]

Stream Classification Data Processor.

optional arguments:
  -h, --help            show this help message and exit
  -l, --label           Label for Current Video Extraction
  -v, --video           Video to Extract Frames From
  -t, --target          Target Directory to Extract Frames
  -sk, --skip           Frames to Skip During Extraction
  -q, --quality         Quality of Storing Image ( 1 - 100 )
  -tw, --target_width   Target Frame Width
  -th, --target_height  Target Frame Height
```

## <a name="annotation_processor">Annotation Based File Processor

Annotation based file processor is used to convert video file into training data, against given annotation file. This [script][abfp] takes following arguments as input:

```bash
usage: processor.py [-h] -l LABEL [-v VIDEO] [-a ANNOTATION] [-t TARGET]
                    [-sk SKIP] [-q QUALITY] [-tw TARGET_WIDTH]
                    [-th TARGET_HEIGHT] [-dc DIR_COLUMN]

Stream Classification Data Processor.

optional arguments:
  -h, --help            show this help message and exit
  -l, --label           Label for Current Video Extraction
  -v, --video           Video to Extract Frames From
  -a, --annotation      Annotation File Against Respective Video
  -t, --target          Target Directory to Extract Frames
  -sk, --skip           Frames to Skip During Extraction
  -q, --quality         Quality of Storing Image ( 1 - 100 )
  -tw, --target_width   Target Frame Width
  -th, --target_height  Target Frame Height
  -dc, --dir_column     CSV Column to Make Class Directories From
```

### <a name="stream_annotation_format">Stream Object Annotation Format

In case of ***Annotation Based File Processor***, following annotation format should be followed in ***.csv*** file:

| Sr.No. |  Stream type  |   Brand Name  |    Description    | End_H | End_M | End_S | End_Cut |
|:------:|:-------------:|:-------------:|:-----------------:|:-----:|:-----:|:-----:|:-------:|
|    1   | Advertisement |   Innovative  |      Biscuit      |   0   |   0   |   2   |    4    |
|    2   | Advertisement | Tapal Danedar |        Tea        |   0   |   0   |   22  |    13   |
|    3   | Advertisement |     Ensure    |    Milk Powder    |   0   |   0   |   32  |    12   |
|    4   | Advertisement |      Jazz     | Telecommunication |   0   |   0   |   47  |    11   |
|    5   |       --      |       --      |         --        |   --  |   --  |   --  |    --   |
|    6   |       --      |       --      |         --        |   --  |   --  |   --  |    --   |

In the above file / table structure:

* ***Sr.No.*** :  denotes serial number of respective entry.
* ***Stream type*** : denotes type of stream, which we want to classify.
* ***Brand Name*** : denotes optional brand name of subject stream chunk / clip.
* ***Description*** : denotes optional description of subject stream chunk / clip.
* ***End_H*** : denotes accumulated hour at which a particular stream chunk / clip ended.
* ***End_M*** : denotes accumulated minute of the hour, at which particular stream / chunk ended.
* ***End_S*** : denotes accumulated second of minute, at which particular stream / chunk ended.
* ***End_Cut*** : denotes frame cut number of particular second, at which particular stream / chunk ended.

## <a name="trainer">Trainer

Training on the subject data can be automatically done and handled by training pipeline. This [script][trainer] takes following arguments as input:

```bash
usage: trainer.py [-h] -e EPOCHS [-l LR] [-bs BATCH_SIZE]
                  [-trs TRAINING_SPLIT] [-vas VALIDATION_SPLIT]
                  [-tes TESTING_SPLIT] [-sd SEED] [-ts TRAIN_SHUFFLE]
                  [-d DATA] [-ohe OHE] [-ms MSADDR]

Stream Classification Model Trainer.

optional arguments:
  -h, --help                show this help message and exit
  -e, --epochs              Number of Epochs to Which Model Should be Trained Upto
  -l, --lr                  Learning Rate for Model Training
  -bs, --batch_size         Batch Size of Input Data
  -trs, --training_split    Training Split Percentage
  -vas, --validation_split  Validation Split Percentage
  -tes, --testing_split     Testing Split Percentage
  -sd, --seed               Seed Value to Randomize Dataset
  -ts, --train_shuffle      Boolean Valiable to Shuffle Training Data ( True / False )
  -d, --data                Absolute Aaddress of the Parent Directory of Images Sub-Directories
  -ohe, --OHE               Absolute Address to Save One Hot Encoded Labels file
  -ms, --msaddr             Absolute Address to Save Model File
```

[sfp]: ./single_processor.py
[abfp]: ./processor.py
[trainer]: ./trainer.py
