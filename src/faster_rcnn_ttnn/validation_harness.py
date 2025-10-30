import torch, torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def get_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model


def run_cpu_sanity():
    x = torch.rand(3,800,800).unsqueeze(0)
    model = get_model()
    with torch.no_grad():
        out = model(x)[0]
    print("keys:", list(out.keys()))
    for k in ("boxes","labels","scores"):
        if k in out:
            print(k, out[k].shape)


if __name__ == "__main__":
    run_cpu_sanity()
