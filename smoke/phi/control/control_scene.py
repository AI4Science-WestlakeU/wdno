import json, os, inspect, shutil
from phi.fluidformat import *


class ControlScene:

    def __init__(self, path, mode="r", index=None):
        self.path = path
        self.index = index
        if mode.lower() == "r":
            with open(os.path.join(path, "description.json"), "r") as file:
                self.infodict = json.load(file)
        elif mode.lower() == "w":
            self.infodict = {}
        else:
            raise ValueError("Illegal mode: %s " %mode)

    def get_final_loss(self, include_reg_loss=True):
        final_loss = self.infodict["final_loss"]
        if not include_reg_loss and "regloss" in self.infodict:
            final_loss -= self.infodict["regloss"]
        return final_loss

    def improvement(self):
        final_loss = self.get_final_loss(include_reg_loss=False)
        initial_loss = self.infodict["initial_loss"]
        return initial_loss / final_loss

    @property
    def scenetype(self):
        return self.infodict["scenetype"]

    def control_frames(self):
        return range(self.infodict["n_frames"])

    def target_density(self):
        return read_sim_frames(self.path, ["target density"])[0]

    def get_state(self, index):
        return read_sim_frame(self.path, ["density", "velocity", "force"], index, set_missing_to_none=False)

    def time_to_keyframe(self, index):
        return self.infodict["n_frames"] - index

    def put(self, dict, save=True):
        self.infodict.update(dict)
        if save:
            with open(os.path.join(self.path, "description.json"), "w") as out:
                json.dump(self.infodict, out, indent=2)

    def file(self, name):
        return os.path.join(self.path, name)

    def __getitem__(self, key):
        return self.infodict[key]

    def __getattr__(self, item):
        return self.infodict[item]

    def __str__(self):
        return self.path

    def copy_calling_script(self):
        script_path = inspect.stack()[1][1]
        script_name = os.path.basename(script_path)
        src_path = os.path.join(self.path, "src")
        os.path.isdir(src_path) or os.mkdir(src_path)
        target = os.path.join(self.path, "src", script_name)
        shutil.copy(script_path, target)
        try:
            shutil.copystat(script_path, target)
        except:
            pass # print("Could not copy file metadata to %s"%target)


def list_scenes(directory, category, min=None, max=None):
    scenes = []
    if min is None:
        i = 1
    else:
        i = int(min)
    while True:
        path = os.path.join(directory, category, "sim_%06d/"%i)
        if not os.path.isdir(path): break
        scenes.append(ControlScene(path, "r", i))
        if max is not None and i == max: break
        i += 1
    return scenes


def new_scene(directory, category):
    scenedir = os.path.join(directory, category)
    if not os.path.isdir(scenedir):
        os.makedirs(scenedir)
        next_index = 1
    else:
        indices = [int(name[4:]) for name in os.listdir(scenedir) if name.startswith("sim_")]
        if not indices:
            next_index = 1
        else:
            next_index = max(indices) + 1
    path = os.path.join(scenedir, "sim_%06d"%next_index)
    os.mkdir(path)
    return ControlScene(path, "w", next_index)




def load_scene_data(scenes):
    densities = []
    velocities = []
    forces = []
    targets = []
    remaining_times = []

    for scene in scenes:
        target = scene.target_density()
        for i in scene.control_frames():
            density, velocity, force = scene.get_state(i)
            remaining_time = scene.time_to_keyframe(i)
            densities.append(density)
            velocities.append(velocity)
            forces.append(force)
            remaining_times.append(remaining_time)
            targets.append(target)

    return densities, velocities, forces, targets, remaining_times