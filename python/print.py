import sys
print(sys.path[:3])  # 상위 3개만 출력
from pygicp import FastVGICPCuda
from pygicp import FastVGICPCuda
reg = FastVGICPCuda()  # 에러 없이 생성되면 GPU 지원 정상
