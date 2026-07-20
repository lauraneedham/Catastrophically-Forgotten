def evaluate_accuracy(MLP, loader):
    """Computes accuracy of MLP on a given dataloader, without training on it."""
    MLP.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            y_pred = MLP(X)
            pred = torch.argmax(y_pred, axis=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return 100 * correct / total


OLD_CLASSES = [0, 1, 2, 3, 4, 5]
NEW_CLASSES = [6, 7, 8, 9]

train_set_old = restrict_classes(train_set, OLD_CLASSES)
valid_set_old = restrict_classes(valid_set, OLD_CLASSES)
train_set_new = restrict_classes(train_set, NEW_CLASSES)
valid_set_new = restrict_classes(valid_set, NEW_CLASSES)

train_loader_old = torch.utils.data.DataLoader(train_set_old, batch_size=BATCH_SIZE, shuffle=True)
valid_loader_old = torch.utils.data.DataLoader(valid_set_old, batch_size=BATCH_SIZE, shuffle=False)
train_loader_new = torch.utils.data.DataLoader(train_set_new, batch_size=BATCH_SIZE, shuffle=True)
valid_loader_new = torch.utils.data.DataLoader(valid_set_new, batch_size=BATCH_SIZE, shuffle=False)

train_set_full_restricted = restrict_classes(train_set, OLD_CLASSES + NEW_CLASSES)
train_loader_full = torch.utils.data.DataLoader(train_set_full_restricted, batch_size=BATCH_SIZE, shuffle=True)


def run_forgetting_experiment(model_type="backprop", condition="sequential",
                              num_epochs_phase1=6, num_epochs_phase2=6, lr=None):
    model_params = dict(num_hidden=NUM_HIDDEN, num_outputs=len(OLD_CLASSES) + len(NEW_CLASSES), bias=BIAS)

    if model_type == "backprop":
        MLP = MultiLayerPerceptron(**model_params)
        lr = LR if lr is None else lr
    else:
        raise ValueError(f"{model_type} not implemented yet -- add feedback alignment / Kolen-Pollack here next.")

    if isinstance(lr, list):
        optimizer = BasicOptimizer([
            {"params": MLP.lin1.parameters(), "lr": lr[0]},
            {"params": MLP.lin2.parameters(), "lr": lr[1]},
        ])
    else:
        optimizer = BasicOptimizer(MLP.parameters(), lr=lr)

    phase1_results = train_model(
        MLP, train_loader_old, valid_loader_old, optimizer,
        num_epochs=num_epochs_phase1,
    )

    phase2_train_loader = train_loader_new if condition == "sequential" else train_loader_full
    phase2_results = train_model(
        MLP, phase2_train_loader, valid_loader_old, optimizer,
        num_epochs=num_epochs_phase2,
    )

    old_class_acc_trace = (
        phase1_results["avg_valid_accuracies"] + phase2_results["avg_valid_accuracies"]
    )
    new_class_acc_final = evaluate_accuracy(MLP, valid_loader_new)

    return MLP, old_class_acc_trace, new_class_acc_final, num_epochs_phase1


results = {}
epoch_settings = {
    "backprop": dict(num_epochs_phase1=6, num_epochs_phase2=6, lr=LR),
}

for model_type in ["backprop"]:
    for condition in ["sequential", "interleaved"]:
        _, old_trace, new_final, intro_epoch = run_forgetting_experiment(
            model_type=model_type, condition=condition, **epoch_settings[model_type]
        )
        results[f"{model_type}_{condition}"] = (old_trace, new_final, intro_epoch)
        print(f"{model_type:>8} / {condition:<11}: "
              f"acc on 0-5 right before intro = {old_trace[intro_epoch-1]:5.1f}%, "
              f"after phase 2 = {old_trace[-1]:5.1f}%, "
              f"final acc on 6-9 = {new_final:5.1f}%")



fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
colors = {"sequential": "#a15c00", "interleaved": "#2c5282"}

for ax, model_type in zip(axes, ["backprop"]):
    for condition in ["sequential", "interleaved"]:
        old_trace, new_final, intro_epoch = results[f"{model_type}_{condition}"]
        ax.plot(range(1, len(old_trace) + 1), old_trace, label=condition, color=colors[condition])
    ax.axvline(intro_epoch + 0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(model_type.capitalize())
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy on digits 0-5 (%)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()