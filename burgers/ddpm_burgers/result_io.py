import yaml
import numpy as np
import torch
from pathlib import Path
import os
import copy

class YamlReaderError(Exception):
    pass


def data_merge(a, b) -> None:
    """merges b into a and return merged result
    NOTE: side effect: a is modified
    NOTE: tuples and arbitrary objects are not handled as it is totally ambiguous what should happen
    """
    key = None
    # ## debug output
    # sys.stderr.write("DEBUG: %s to %s\n" %(b,a))
    try:
        if (
            a is None
            or isinstance(a, str)
            or isinstance(a, int)
            or isinstance(a, float)
        ):
            # border case for first run or if a is a primitive
            a = b
        elif isinstance(a, list):
            # lists can be only appended
            if isinstance(b, list):
                # merge lists
                a.extend(b)
            else:
                # append to list
                a.append(b)
        elif isinstance(a, dict):
            # dicts must be merged
            if isinstance(b, dict):
                for key in b:
                    if key in a:
                        a[key] = data_merge(a[key], b[key])
                    else:
                        a[key] = b[key]
            else:
                raise YamlReaderError(
                    'Cannot merge non-dict "%s" into dict "%s"' % (b, a)
                )
        else:
            raise YamlReaderError('NOT IMPLEMENTED "%s" into "%s"' % (b, a))
    except TypeError as e:
        raise YamlReaderError(
            'TypeError "%s" in key "%s" when merging "%s" into "%s"' % (e, key, b, a)
        )
    return a

def merge_save_dict(fname, new_res):
    # save to file
    # creates file if not exist
    directory = os.path.dirname(fname)
    if directory and not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(fname):
        with open(fname, 'w'): pass

    with open(fname, "r") as f:
        res = yaml.safe_load(f)

    res = data_merge(res, new_res)  # join recursively result dicts

    with open(fname, "w") as f:
        yaml.dump(res, f)

def save_acc(
        acc, 
        fname,
        make_dict_path: callable=lambda acc_dict, dict_args:{dict_args['model_name']: {"result": acc_dict}}, 
        **dict_path_args
    ):
    # NOTE: convert to python float
    new_res = make_dict_path({"mean": float(np.mean(acc)), "std": float(np.std(acc))}, dict_path_args)
    merge_save_dict(fname, new_res)


def save_edge_flip(edge_flip, fpath):

    edge_flip = edge_flip.detach().cpu().numpy()
    # check if path exists, if not, create that directory.
    Path(fpath).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(fpath, 'flip.npy'), edge_flip)


def load_edge_flip(fpath):
    if not os.path.exists(fpath):
        raise ValueError('Edge filp file not found: path does not exist')
    try:
        edge_flip = np.load(os.path.join(fpath, 'flip.npy'))
    except Exception as e:
        raise ValueError('Edge filp file not found: unknown error. ' + e)
    return torch.from_numpy(edge_flip).cuda()




def rep_save_model(model_name, budget, models, rep_per_split=5, save_name='clean_model', dataset_name='cora'):
    for rep, model in enumerate(models):
        fpath = f'./my_exp/{dataset_name}_flips/{model_name}/models/{budget:.3f}/split_{rep // rep_per_split}/rand_model_{rep % rep_per_split}/'
        Path(fpath).mkdir(parents=True, exist_ok=True)
        fpath += save_name
        torch.save(model.state_dict(), fpath)

def rep_load_model(model_name, budget, model, rep_per_split=5, save_name='clean_model', neglect_ada_model=False, dataset_name='cora', debug=False):
    models = []
    rep = 0
    while True:
        try:
            if neglect_ada_model and rep % rep_per_split == 0:
                raise Exception # not load the model init that is attacked
            fpath = f'{os.path.dirname(__file__)}/{dataset_name}_flips/{model_name}/models/{budget:.3f}/split_{rep // rep_per_split}/rand_model_{rep % rep_per_split}/{save_name}'
            cur_model: torch.nn.Module = copy.deepcopy(model)
            cur_model.load_state_dict(torch.load(fpath))
            models.append(cur_model.eval())
        except Exception as e:
            if debug:
                print(e)
            break
        rep += 1
    return models





def save_rep_edge_flips(model_name, budget: float, attack_name, flip_ls, dataset_name='cora'):
    for rep, flip in enumerate(flip_ls):
        if type(flip) is list:
            for i, f in enumerate(flip):
                fpath = f'./my_exp/{dataset_name}_flips/{model_name}/{attack_name}/{budget:.3f}/split_{rep}/node_{i}/'
                save_edge_flip(f, fpath)
        else:
            # save flip to file
            fpath = f'./my_exp/{dataset_name}_flips/{model_name}/{attack_name}/{budget:.3f}/split_{rep}/'
            save_edge_flip(flip, fpath)


def load_rep_edge_flips(model_name, budget: float, attack_name, dataset_name='cora'):
    flips = []
    rep = 0
    while True:
        try:
            if 'global' in attack_name:
                fpath = f'{os.path.dirname(__file__)}/{dataset_name}_flips/{model_name}/{attack_name}/{budget:.3f}/split_{rep}/'
                flip = load_edge_flip(fpath)
                flips.append(flip)
            elif 'local' in attack_name:
                cur_flips = []
                node_idx = 0
                while True:
                    fpath = f'{os.path.dirname(__file__)}/{dataset_name}_flips/{model_name}/{attack_name}/{budget:.3f}/split_{rep}/'
                    
                    if os.path.exists(fpath) and not os.path.exists(fpath + f'node_{node_idx}/'):
                        print(f'{node_idx} nodes loaded')
                        break

                    fpath += f'node_{node_idx}/'
                    flip = load_edge_flip(fpath)
                    cur_flips.append(flip)
                    node_idx += 1
                    
                flips.append(cur_flips)

        except Exception as e:
            print('Loading terminates.', f'{rep} splits loaded.')
            if rep == 0:
                print('Last error message:', e)
            break
        rep += 1
    return flips
        
