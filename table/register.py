import world
import dataloader
import model
import utils
from pprint import pprint

if world.model_name == 'lgn':
    dataset = dataloader.Loader(path=""+world.dataset)
elif world.model_name == 'Ours':
    dataset = dataloader.Loadernew(path="" + world.dataset)
elif world.model_name == 'mf':
    dataset = dataloader.LastFM()

# print('===========config================')
# pprint(world.config)
# print("cores for test:", world.CORES)
# print("comment:", world.comment)
# print("tensorboard:", world.tensorboard)
# print("LOAD:", world.LOAD)
# print("Weight path:", world.PATH)
# print("Test Topks:", world.topks)
# print("using bpr loss")
# print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'Ours':model.Ours,
}