# Sora-Generates-Videos-with-Stunning-Geometrical-Consistency
Sora Generates Videos with Stunning Geometrical Consistency
## software
### software whl download link:
https://drive.google.com/file/d/1scoJ-mLoZ_3ZkrALQfdwwnw-0Q30NB5e/view?usp=drive_link
### install:
pip install GVGC-0.0.1-py3-none-any.whl

### usage:
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

