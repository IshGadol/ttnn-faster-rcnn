import os, json, glob, csv, time, argparse
import torch, torchvision
import cv2
from torchvision.transforms.functional import to_tensor
from torchvision.ops import nms

def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_csv_row(csv_path, header, row):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)

def safe_crop(img, box):
    """Crop with boundary checks. Returns None if invalid."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def main():
    ap = argparse.ArgumentParser(description="Run Faster R-CNN on images and log results.")
    ap.add_argument("--images-dir", default="images", help="Directory of input images")
    ap.add_argument("--out-dir", default="outputs", help="Output directory for preds/vis")
    ap.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold for drawing/metrics")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--save-crops", action="store_true", help="Save cropped detections per image")
    ap.add_argument("--max-crops-per-image", type=int, default=50, help="Hard cap on crops saved per image")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir("reports")

    img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")) +
                       glob.glob(os.path.join(args.images_dir, "*.png")) +
                       glob.glob(os.path.join(args.images_dir, "*.jpeg")))

    if not img_paths:
        print(f"No images found in ./{args.images_dir} — add a few .jpg/.png and retry.")
        return

    model = load_model()

    # CSV log
    csv_path = "reports/infer_log.csv"
    header = ["image", "elapsed_ms", "detections_kept", "mean_score_kept", "min_score_kept", "max_score_kept", "crops_saved", "notes"]

    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]

        # Robust load
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print(f"[WARN] Skipping unreadable image: {p}")
            append_csv_row(csv_path, header, [os.path.basename(p), "", 0, "", "", "", 0, "unreadable"])
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = to_tensor(img_rgb).unsqueeze(0)

        # Inference + timing
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = out.get("boxes", torch.empty((0,4)))
        labels = out.get("labels", torch.empty((0,), dtype=torch.int64))
        scores = out.get("scores", torch.empty((0,)))

        # NMS
        if boxes.numel() > 0 and scores.numel() > 0:
            keep = nms(boxes, scores, args.nms_iou)
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        # Thresholded detections
        if scores.numel() > 0:
            keep_thr = scores >= args.score_thresh
            k_boxes = boxes[keep_thr]
            k_scores = scores[keep_thr]
        else:
            k_boxes = torch.empty((0,4))
            k_scores = torch.empty((0,))

        # JSON output (all NMS-filtered detections, not only thresholded)
        jpath = os.path.join(args.out_dir, f"{base}_preds.json")
        write_json(jpath, {
            "image": os.path.basename(p),
            "boxes": boxes.cpu().tolist(),
            "labels": labels.cpu().tolist(),
            "scores": [float(s) for s in scores.cpu().tolist()],
            "nms_iou": args.nms_iou,
            "score_thresh": args.score_thresh,
            "elapsed_ms": round(elapsed_ms, 3),
        })

        # Visualization for kept (thresholded) detections
        vis = img_bgr.copy()
        for (x1,y1,x2,y2), s in zip(k_boxes.cpu().tolist(), k_scores.cpu().tolist()):
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{s:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        vpath = os.path.join(args.out_dir, f"{base}_vis.jpg")
        cv2.imwrite(vpath, vis)

        # Optional crops
        crops_saved = 0
        if args.save_crops and k_boxes.numel() > 0:
            crops_dir = os.path.join(args.out_dir, "crops", base)
            ensure_dir(crops_dir)
            for i, (box, sc) in enumerate(zip(k_boxes.cpu().tolist(), k_scores.cpu().tolist())):
                if i >= args.max_crops_per_image:
                    break
                crop = safe_crop(img_bgr, box)
                if crop is None:
                    continue
                crop_path = os.path.join(crops_dir, f"{i:03d}_score{sc:.2f}.jpg")
                cv2.imwrite(crop_path, crop)
                crops_saved += 1

        # CSV logging
        if k_scores.numel() > 0:
            mean_s = float(k_scores.mean().item())
            min_s  = float(k_scores.min().item())
            max_s  = float(k_scores.max().item())
            kept   = int(k_scores.numel())
        else:
            mean_s = min_s = max_s = ""
            kept   = 0

        append_csv_row(
            csv_path, header,
            [os.path.basename(p), round(elapsed_ms, 3), kept, mean_s, min_s, max_s, crops_saved, ""]
        )

        print(f"Processed {p} -> {jpath}, {vpath}  |  {kept} kept @ ≥{args.score_thresh}  |  crops_saved={crops_saved}")

if __name__ == "__main__":
    main()
