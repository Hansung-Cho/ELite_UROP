import sys
sys.path.insert(0, "../build")   # build 폴더가 python/ 바로 위니까
import elite_utils

pts = [elite_utils.Point(0,0,0),
       elite_utils.Point(1,1,5),
       elite_utils.Point(2,2,10)]
out = elite_utils.height_filter(pts, 1.0)
print("Filtered z’s:", [p.z for p in out])