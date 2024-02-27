# Sora-Generates-Videos-with-Stunning-Geometrical-Consistency
Sora Generates Videos with Stunning Geometrical Consistency
## data
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

## software (python)
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

