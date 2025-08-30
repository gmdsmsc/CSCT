import torch.nn as nn

def count_leaf_layers(model):
    return sum(1 for m in model.modules() if len(list(m.children())) == 0)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def layerwise_summary(model):
    summary = []
    for name, param in model.named_parameters():
        summary.append({
            "name": name,
            "shape": tuple(param.shape),
            "requires_grad": param.requires_grad,
            "count": param.numel()
        })
    return summary

def print_model_diagnostics(model, model_name="UnnamedModel"):
    print(f"\n📊 Model Diagnostics: {model_name}")
    print(f"Leaf Layers: {count_leaf_layers(model)}")
    total, trainable = count_parameters(model)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print("\n🔍 Layerwise Breakdown:")
    for entry in layerwise_summary(model):
        print(f"{entry['name']:40} | shape: {str(entry['shape']):20} | trainable: {entry['requires_grad']} | count: {entry['count']:,}")



if __name__ == '__main__':
    from configs.flappy.PredFormer import model_config
    from openstl.models.TSST_baseline_film import PredFormer_Model
    model = PredFormer_Model(model_config)
    print_model_diagnostics(model, model_name="PredFormer_Model")

    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # Dummy input: (batch_size, frames, channels, height, width)
    input = torch.randn(1, 10, 1, 84, 84)
    label_x = torch.randint(0, 1, (1, 10)).long()
    label_y = torch.randint(0, 1, (1, 10)).long()

    # FLOPs
    flops = FlopCountAnalysis(model, (input, label_x, label_y))
    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

    # Parameters
    print(parameter_count_table(model))
