import numpy as np
import open3d as o3d
from alignment.matcher.base_scan_matcher import BaseScanMatcher
from pygicp import FastVGICPCuda  # GPU 전용

class FastGICPScanMatcher(BaseScanMatcher):
    def __init__(
        self,
        max_correspondence_distance: float = 1.0,
        init_transformation: np.ndarray = None,
        voxel_resolution: float = 1.0,
        max_iter: int = 64,
    ):
        super().__init__(max_correspondence_distance, init_transformation)

        # 무조건 GPU 버전 사용
        self.registration = FastVGICPCuda()
        self.registration.set_resolution(voxel_resolution)
        # FastVGICPCuda에는 set_max_iterations가 없으므로 제거
        self.registration.set_correspondence_randomness(15)

    def align(self) -> None:
        if self.source_raw is None or self.target_raw is None:
            raise RuntimeError("Source/target not set.")

        src = np.asarray(self.source_raw.points, dtype=np.float32)
        tgt = np.asarray(self.target_raw.points, dtype=np.float32)

        # FastVGICPCuda는 src/tgt를 세팅하고 align에는 initial_guess만 전달
        self.registration.set_input_target(tgt)
        self.registration.set_input_source(src)

        # align 호출
        self.transformation = self.registration.align(self.init_transformation.astype(np.float32))

        # 변환된 소스 포인트 계산
        pts_h = np.hstack((src, np.ones((src.shape[0], 1))))
        transformed = (self.transformation @ pts_h.T).T[:, :3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(transformed)
        self.transformed = pc