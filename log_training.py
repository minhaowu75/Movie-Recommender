import sys

def log(epoch, step, total_loss, log_progress_step, data_size, losses, EPOCHS):
    avg_loss = total_loss/log_progress_step
    sys.stderr.write(
        f"\r{epoch+1:02d}/{EPOCHS:02d} | Step: {step}/{data_size} | Avg Loss: {avg_loss:<6.9f}"
    )
    sys.stderr.flush()
    losses.append(avg_loss)