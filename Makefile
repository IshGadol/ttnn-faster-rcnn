.PHONY: env sanity tt-demo


env:
	python -m pip install -r requirements.txt


sanity:
	python -m src.faster_rcnn_ttnn.validation_harness


tt-demo:
	python scripts/ttnn_device_check.py || true

images-bench:
	python scripts/run_images_cpu.py --score-thresh 0.5

images-crops:
	python scripts/run_images_cpu.py --score-thresh 0.5 --save-crops --max-crops-per-image 50
