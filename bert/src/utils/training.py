def custom_lr_schedule(
    step_no: int, warm_up: int = 10000, total_steps: int = 100000
):
    post_warm_up_steps = total_steps - warm_up
    if step_no == 0:
        step_no = 1
    if step_no <= warm_up:
        return step_no / warm_up
    else:
        return (total_steps - step_no) / post_warm_up_steps
    
class TrainState:
    n_steps: int = 0
    n_updates: int = 0
    train_loss = {"step": [], "epoch": []}
    val_loss = {"step": [], "epoch": []}
    lr = []