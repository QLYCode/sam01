import torch

import utils.helpers as helpers
import utils.metrics as metrics

def run_epoch(data_loader, model, args, writer, loss, best_loss, iter_num, device):
    model.eval()
    for sampled_batch in data_loader:
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch = volume_batch.to(device)
        label_batch = label_batch.to(device)
        helpers.check_for_nan(volume_batch)
        with torch.no_grad():
            outputs = model(volume_batch)
        outputs_soft = torch.softmax(outputs.clone(), dim=1)
        loss_ce = loss(outputs, label_batch[:].long())
        total_loss = loss_ce
        hausdorff, assd = metrics.calculate_distances(outputs_soft.clone(), label_batch)
        dice = metrics.dice_score(outputs_soft.clone(), label_batch)
        scalars = {
            "total_loss": total_loss,
            "dice": dice,
            "hausdorff": hausdorff,
            "assd": assd
        }
        helpers.log_epoch("val", writer, iter_num, scalars, volume_batch, outputs_soft, label_batch, None)
        
        if total_loss.item() < best_loss:
            torch.save(model.state_dict(), f"{helpers.get_snapshot_path(args)}/best.pth")
            print(f"saved new best model (loss={total_loss}, dice={dice})")
            best_loss = total_loss.item()

        iter_num += 1
        # if iter_num >= args.max_iterations:
        if iter_num > 10:
            break
    return iter_num, best_loss
        