# Sora-Generates-Videos-with-Stunning-Geometrical-Consistency
Generates Videos  Geometrical Consistency metric (GVGC metric)
## Homepase
https://sora-geometrical-consistency.github.io/
## Fast forward


## Data
### data download link:
https://drive.google.com/file/d/1E_7DR_DIvvWtDXn5KXUwXfBIA_3MhBMG/view?usp=drive_link
place the data file as fllows:
```python
root
|
---xxx.py
---XXX.py
---data
   |
   ---sora
   |
   ---gen2
   |
   ---pika
```
## Code
### code usage
```python
python Eval_all.py
```
## Software
### software whl download link:
https://drive.google.com/file/d/1scoJ-mLoZ_3ZkrALQfdwwnw-0Q30NB5e/view?usp=drive_link
### install:
```python
pip install GVGC-0.0.1-py3-none-any.whl
```

### software usage:
```python
#!/usr/bin/env python3
from AutoExtraFrame import AutoExtraFrame
from PatchAutoEvaluate import PatchAutoEvaluate
from PatchDenseMatch import PatchDenseMatch
from PatchDrawMatch import PatchDrawMatch

video_list = ["sora", "pika", "gen2"]
AutoExtraFrame(video_list)
PatchAutoEvaluate(video_list)
PatchDenseMatch(video_list)
PatchDrawMatch(video_list)
```
## Result
```python
```python
root
|
---xxx.py
---XXX.py
---data
   |
   ---sora
   |
   ---------brief_result
   ---------full_result
   ---------image_result
   |
   ---gen2
   |
   ---------brief_result
   ---------full_result
   ---------image_result
   |
   ---pika
   |
   ---------brief_result
   ---------full_result
   ---------image_result
```
```

