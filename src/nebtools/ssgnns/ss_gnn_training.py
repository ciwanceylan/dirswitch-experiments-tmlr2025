from typing import List
import time
import nebtools.ssgnns.utils as sgnutl


def train_ss_gnn(
    model_trainer: sgnutl.SSGNNTrainer,
    num_epochs: int,
    eval_callbacks: List[sgnutl.TestEvalCallback],
    eval_every: int = 10,
    verbose: bool = False,
    get_slopes_callback=None,
):
    loss_history = []
    train_times = []
    score_histories = {eval_cb.name: [] for eval_cb in eval_callbacks}
    slopes_history = []

    for epoch in range(num_epochs):
        if epoch % eval_every == 0:
            if verbose:
                print("Evaluating...")
            if get_slopes_callback is not None:
                slopes_history.append(
                    {"epoch": epoch, "slopes": get_slopes_callback(model_trainer)}
                )
            for eval_cb in eval_callbacks:
                start = time.perf_counter()
                score_histories[eval_cb.name] += sgnutl.get_evaluation(
                    model_trainer, eval_cb, epoch=epoch
                )
                duration = time.perf_counter() - start
                if verbose:
                    print(f"Eval {eval_cb.name} done after {duration} sec.")
        start = time.perf_counter()
        loss = model_trainer.step(step=epoch)
        train_times.append(time.perf_counter() - start)
        loss_history.append(loss)

        if verbose:
            print(f"Epoch={epoch:03d}, loss={loss:.4f}")

    final_scores = dict()
    for eval_cb in eval_callbacks:
        final_scores[eval_cb.name] = sgnutl.get_evaluation(
            model_trainer, eval_cb, epoch=num_epochs
        )

    # Do final evaluation
    return final_scores, score_histories, loss_history, train_times, slopes_history


def train_ss_gnn_without_eval(
    model_trainer: sgnutl.SSGNNTrainer, num_epochs: int, verbose: bool
):
    loss_history = []
    for epoch in range(num_epochs):
        loss = model_trainer.step(step=epoch)
        loss_history.append(loss)
        if verbose:
            print(f"Epoch={epoch:03d}, loss={loss:.4f}")
    return model_trainer, loss_history
