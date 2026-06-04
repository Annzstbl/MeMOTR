import cv2
import numpy as np
import torch
from typing import Iterable, List, Optional

from hsmot.mmlab.hs_mmrotate import obb2poly_np, poly2obb_np


def _build_detection_mask(height: int, width: int, detections, downscale: int) -> np.ndarray:
    """Create a mask that suppresses regions covered by detections."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
    if detections is None:
        return mask

    dets = np.asarray(detections, dtype=np.float32)
    if dets.ndim == 1:
        dets = dets[None, :]
    for det in dets:
        tlbr = (det[:4] / downscale).astype(int)
        x0 = np.clip(tlbr[0], 0, width)
        y0 = np.clip(tlbr[1], 0, height)
        x1 = np.clip(tlbr[2], 0, width)
        y1 = np.clip(tlbr[3], 0, height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    return mask


def _estimate_affine(prev_pts: np.ndarray, curr_pts: np.ndarray) -> np.ndarray:
    """Estimate affine transform from point correspondences."""
    if prev_pts.shape[0] < 4:
        return np.eye(2, 3, dtype=np.float32)
    H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
    if H is None:
        return np.eye(2, 3, dtype=np.float32)
    return H.astype(np.float32)


def _compute_orb_or_sift_sequence(frames: List[np.ndarray],
                                  method: str,
                                  detections_list: List,
                                  downscale: int) -> List[np.ndarray]:
    if method == 'orb':
        detector = cv2.FastFeatureDetector_create(20)
        extractor = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
        extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    gmcs = [np.eye(2, 3, dtype=np.float32)]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    if downscale > 1:
        prev_gray = cv2.resize(prev_gray, (prev_gray.shape[1] // downscale, prev_gray.shape[0] // downscale))
    mask0 = _build_detection_mask(prev_gray.shape[0], prev_gray.shape[1], detections_list[0], downscale)
    prev_kp = detector.detect(prev_gray, mask0)
    prev_kp, prev_desc = extractor.compute(prev_gray, prev_kp)

    for idx in range(1, len(frames)):
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // downscale, gray.shape[0] // downscale))
        mask = _build_detection_mask(gray.shape[0], gray.shape[1], detections_list[idx], downscale)
        kp = detector.detect(gray, mask)
        kp, desc = extractor.compute(gray, kp)

        H = np.eye(2, 3, dtype=np.float32)
        if prev_desc is not None and desc is not None and len(prev_desc) > 0 and len(desc) > 0:
            knn_matches = matcher.knnMatch(prev_desc, desc, k=2)
            matches = []
            for m, n in knn_matches:
                if m.distance < 0.9 * n.distance:
                    matches.append(m)
            if len(matches) >= 4:
                prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                curr_pts = np.float32([kp[m.trainIdx].pt for m in matches])
                H = _estimate_affine(prev_pts, curr_pts)
                if downscale > 1:
                    H[0, 2] *= downscale
                    H[1, 2] *= downscale

        gmcs.append(H)
        prev_gray = gray
        prev_kp, prev_desc = kp, desc
    return gmcs


def _compute_sparse_optical_flow_sequence(frames: List[np.ndarray],
                                          downscale: int) -> List[np.ndarray]:
    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                          useHarrisDetector=False, k=0.04)
    gmcs = [np.eye(2, 3, dtype=np.float32)]

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    if downscale > 1:
        prev_gray = cv2.resize(prev_gray, (prev_gray.shape[1] // downscale, prev_gray.shape[0] // downscale))
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    for idx in range(1, len(frames)):
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // downscale, gray.shape[0] // downscale))

        if prev_pts is not None and len(prev_pts) >= 4:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            valid_prev = prev_pts[status.flatten() == 1]
            valid_curr = curr_pts[status.flatten() == 1]
            H = _estimate_affine(valid_prev, valid_curr)
        else:
            H = np.eye(2, 3, dtype=np.float32)

        if downscale > 1:
            H[0, 2] *= downscale
            H[1, 2] *= downscale
        gmcs.append(H)

        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    return gmcs


def _compute_ecc_sequence(frames: List[np.ndarray],
                          downscale: int) -> List[np.ndarray]:
    gmcs = [np.eye(2, 3, dtype=np.float32)]
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    if downscale > 1:
        prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 1.5)
        prev_gray = cv2.resize(prev_gray, (prev_gray.shape[1] // downscale, prev_gray.shape[0] // downscale))

    for idx in range(1, len(frames)):
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
            gray = cv2.resize(gray, (gray.shape[1] // downscale, gray.shape[0] // downscale))

        H = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(prev_gray, gray, H, warp_mode, criteria, None, 1)
        except cv2.error:
            H = np.eye(2, 3, dtype=np.float32)
        gmcs.append(H)
        prev_gray = gray
    return gmcs


def compute_gmc_transform(frame_t0: np.ndarray,
                          frame_t1: np.ndarray,
                          method: str = 'sparseOptFlow',
                          detections=None,
                          downscale: int = 1) -> np.ndarray:
    """
    Compute the global motion compensation affine matrix between two frames.

    Args:
        frame_t0: Previous frame (H, W, 3) in BGR.
        frame_t1: Current frame (H, W, 3) in BGR.
        method: 'sparseOptFlow', 'orb', 'sift', 'ecc', or 'none'.
        detections: Optional list/array of boxes [x1, y1, x2, y2] to suppress when detecting features.
        downscale: Integer downscale factor for faster computation.

    Returns:
        2x3 affine matrix that warps frame_t0 to frame_t1.
    """
    method = method.lower()
    downscale = max(1, int(downscale))

    if method == 'none':
        return np.eye(2, 3, dtype=np.float32)
    frames = [frame_t0, frame_t1]
    detections_list = [detections, detections]
    gmcs = _dispatch_gmc_sequence(frames, method, detections_list, downscale)
    return gmcs[-1]


def _warp_poly8(poly8: np.ndarray, gmc_matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly8[:8], dtype=np.float32).reshape(4, 2)
    pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
    return (gmc_matrix @ pts_h.T).T.reshape(-1)


def compensate_rect_boxes(boxes: np.ndarray, gmc_matrix: np.ndarray) -> np.ndarray:
    """
    Apply GMC affine matrix to axis-aligned boxes in pixel cxcywh.

    Args:
        boxes: (N, 4) as [cx, cy, w, h] in pixel coordinates.
        gmc_matrix: 2x3 affine matrix (warps previous frame coords to current).

    Returns:
        Transformed boxes with the same format.
    """
    if boxes is None or len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]

    out = []
    for box in boxes:
        cx, cy, w, h = box
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32,
        )
        warped = _warp_poly8(corners.reshape(-1), gmc_matrix).reshape(4, 2)
        wx1, wy1 = warped[:, 0].min(), warped[:, 1].min()
        wx2, wy2 = warped[:, 0].max(), warped[:, 1].max()
        out.append([(wx1 + wx2) * 0.5, (wy1 + wy2) * 0.5, wx2 - wx1, wy2 - wy1])
    return np.asarray(out, dtype=np.float32)


def compensate_rotated_boxes(boxes: np.ndarray,
                             gmc_matrix: np.ndarray,
                             version: str = "le135") -> np.ndarray:
    """
    Apply GMC affine matrix to rotated bounding boxes.

    Args:
        boxes: (N, 5) as [cx, cy, w, h, angle] in pixel coords; angle is radians
            for ``le135`` / ``le90`` (same convention as ``rotate_norm_boxes_to_boxes``).
        gmc_matrix: 2x3 affine matrix (warps previous frame coords to current).
        version: OBB angle convention, default ``le135``.

    Returns:
        Transformed boxes with the same format.
    """
    if boxes is None or len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]

    boxes_scored = np.concatenate(
        [boxes, np.ones((boxes.shape[0], 1), dtype=np.float32)], axis=1
    )
    polys = obb2poly_np(boxes_scored, version=version)
    if polys.ndim == 1:
        polys = polys.reshape(1, -1)

    transformed = []
    for i, box in enumerate(boxes):
        warped_poly = _warp_poly8(polys[i], gmc_matrix)
        obb = poly2obb_np(warped_poly, version=version)
        if obb is None:
            transformed.append(box.tolist())
        else:
            transformed.append(list(obb))
    return np.asarray(transformed, dtype=np.float32)


def _tensor8ch_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor of shape [C, H, W] (C >= 1) into a BGR numpy array.
    For C >= 3, the first 3 channels are used; otherwise, the single channel is replicated.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")
    tensor = image_tensor.detach().cpu()
    if tensor.dim() != 3:
        raise ValueError(f"Expect tensor with shape [C, H, W], got {tuple(tensor.shape)}")

    c, h, w = tensor.shape
    if c >= 4:
        # 取指定的3个通道
        data = tensor[[3, 1, 0], :, :]
    elif c >= 3:
        data = tensor[:3]
    else:
        data = tensor[0].unsqueeze(0).repeat(3, 1, 1)

    data = data.permute(1, 2, 0).contiguous().numpy().astype(np.float32)

    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)
    data = (data * 255.0).clip(0, 255).astype(np.uint8)

    return data[..., ::-1]


def _dispatch_gmc_sequence(frames: List[np.ndarray],
                           method: str,
                           detections_list: List,
                           downscale: int) -> List[np.ndarray]:
    method = method.lower()
    if method == 'none':
        return [np.eye(2, 3, dtype=np.float32) for _ in frames]
    if method == 'sparseoptflow':
        return _compute_sparse_optical_flow_sequence(frames, downscale)
    if method == 'orb':
        return _compute_orb_or_sift_sequence(frames, 'orb', detections_list, downscale)
    if method == 'sift':
        return _compute_orb_or_sift_sequence(frames, 'sift', detections_list, downscale)
    if method == 'ecc':
        return _compute_ecc_sequence(frames, downscale)
    raise ValueError(f"Unsupported GMC method: {method}")


def compute_gmc_sequence(images: Iterable[torch.Tensor],
                         method: str = 'sparseOptFlow',
                         detections_list: Optional[List] = None,
                         downscale: int = 1) -> List[np.ndarray]:
    """
    Compute GMC matrices for a list of tensor images.

    Args:
        images: iterable of tensors [C, H, W] (typically C=8).
        method: GMC method name.
        detections_list: optional list matching images to suppress detections.
        downscale: downscale factor for GMC computation.

    Returns:
        List of affine matrices for each frame (first frame is identity).
    """
    images = list(images)
    n = len(images)
    if n == 0:
        return []
    if detections_list is None:
        detections_list = [None] * n
    elif len(detections_list) != n:
        raise ValueError("detections_list must have the same length as images")

    frames = [_tensor8ch_to_bgr(img) for img in images]
    return _dispatch_gmc_sequence(frames, method, detections_list, downscale)