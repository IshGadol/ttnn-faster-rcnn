# Cloud Access Request and Scope Confirmation — Faster-RCNN TTNN Bounty

**To:** @marty1885  
**From:** @IshGadol  
**Date:** October 21, 2025  

---

## Acknowledgment

Thanks for the assignment and confirmation. I acknowledge receipt and will post a progress update within **two weeks (by Tuesday, November 4, 2025, IDT)** per Tenstorrent bounty policy.

---

## Cloud Hardware Access Request

**Purpose:** To conduct reproducible bring-up and performance benchmarking for the Faster-RCNN TTNN bounty stages 1–3.

1. **Hardware Requested**  
   - N150 and/or N300 hardware (preferably both for comparative perf runs).  

2. **Access Method**  
   - SSH key-based login preferred.  
   - If a portal or JupyterLab environment is standard for Tenstorrent bounties, I can adapt to that workflow.  

3. **Environment Setup (if preinstallation possible)**  
   a. `tt-metal` — current stable or bounty-approved branch.  
   b. `ttnn` — matching build to ensure operator compatibility.  
   c. Python 3.10+ with: `numpy`, `pyyaml`, `tqdm`, `matplotlib`.  
   d. (Optional) `torch`, `torchvision` for Stage-1 parity validation.  
   e. Tenstorrent profiler and perf-report utilities.  

4. **Data Access**  
   - Please confirm whether a shared **COCO val2017 (5 k images)** directory exists on the cloud nodes. If not, I will upload a small validation subset (~500 images) to my home path.  

5. **Storage Quota**  
   - Requesting **50–100 GB** working space for logs, small COCO slice, and Stage-2/3 perf artifacts.  

6. **Containerization Preference**  
   - Either bare-metal venv or Docker/Podman is fine. If Docker is used, standard non-privileged build/run rights are sufficient.  

---

## Scope Clarifications (Quick Yes/No)

1. **Stage-1 CPU Fallbacks**  
   Temporary CPU fallback for **ROIAlign** and **NMS** permitted **only for correctness**, not for performance measurement, while feature requests are filed — confirm?  

2. **Input Resolutions**  
   Initial bring-up at **320×320**, then extend to **640×640**. Any canonical pre-processing (e.g., resize vs letterbox) Tenstorrent prefers?  

3. **Throughput Metric**  
   ≥ **10 FPS**, measured as the **median FPS over a 30-second inference window** with **batch = 1** at a single resolution — confirm acceptable definition?  

4. **Performance Report Format**  
   Please confirm current **perf sheet + header** templates/links required for bounty submission.  

5. **TT-CNN Flow Reference**  
   We’ll follow the **TT-CNN sharding/interleave + fused conv+relu** guidance. If there’s an updated internal doc beyond YOLOv4, we’d like to use it.  

---

## Milestone Plan

| Week | Dates | Goals |
|------|--------|-------|
| **Week 1** | Oct 21 – Oct 28 | Stage-1 scaffolding; backbone + FPN + RPN parity vs PyTorch; sample inference; small COCO slice mAP; open **draft PR** with validation logs. |
| **Week 2** | Oct 29 – Nov 4 | Stage-2 baseline perf: sharding/interleave, fused ops, L1 reuse; achieve ≥ 10 FPS; produce initial perf report. |
| **Stretch (Week 3)** | Nov 5 – Nov 11 | Stage-3 deeper optimization at 640²; finalize perf sheet + header; document tuning notes. |

---

**Next Step:**  
Once access credentials or instructions are provided, I’ll connect, verify the environment, and begin Stage-1 bring-up immediately.
