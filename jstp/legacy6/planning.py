import torch

def forecast(model, root, actions=None, horizon=3, wrong=False, prior_weight=0.):
    state = root.expand(18, -1, -1, -1)
    paths, states, readings, proposals, likelihoods, pictures = [], [], [], [], [], {}
    for t in range(horizon):
        own_logits = model.action_head(state).float()
        action = actions[:, t] if actions is not None else (torch.arange(18, device=root.device) if not t else own_logits.argmax(-1))
        likelihoods.append(own_logits.log_softmax(-1).gather(1, action[:, None])[:, 0])
        state = model.transition(state, (action + 6) % 18 if wrong else action, torch.full((18,), .2, device=root.device))
        reading = model.joint(state)
        paths.append(action)
        states.append(state)
        readings.append(reading)
        proposals.append(model.action_head(state))
        if t in (0, horizon - 1):
            pictures[t + 1] = model.decode(state)
    result = {key: torch.stack([r[key] for r in readings], 1) for key in readings[0]}
    consequences = -result["cost"][:, -1] - .15 * result["cost"].mean(1)
    joint_motion_likelihood = torch.stack(likelihoods, 1).mean(1)
    return {**result, "states": states, "actions": torch.stack(paths, 1), "proposals": torch.stack(proposals, 1),
            "images": pictures, "consequence_scores": consequences, "motion_log_likelihood": joint_motion_likelihood,
            "scores": consequences + prior_weight * joint_motion_likelihood}
