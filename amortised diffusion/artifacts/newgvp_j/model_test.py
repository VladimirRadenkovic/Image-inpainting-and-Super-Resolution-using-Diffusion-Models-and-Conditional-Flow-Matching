import torch

old_model = torch.load('model.ckpt')
old_weights = list(old_model['state_dict'].items())

#remove all weights whose name contains schedule
new_weights = [w for w in old_weights if 'schedule' not in w[0]]
print(len(old_weights))
print(len(new_weights))

#save new weights with other keys over model.ckpt as new model ckpt
new_model = old_model
new_model['state_dict'] = dict(new_weights)
torch.save(new_model, 'new_model.ckpt')