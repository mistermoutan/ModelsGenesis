import os
import dill


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_object(o, name, dir_):

    directory = os.path.join(dir_, "{}.pkl".format(name))
    with open(directory, "wb") as f:
        dill.dump(o, f)
    print("SAVED {} object in {}".format(name, dir_))


def load_object(location):

    with open(location, "rb") as f:
        o = dill.load(f)
    print("LOAD {} object".format(location))
    return o


def replace_obj_attributes(config_object, **kwargs):

    for key, value in kwargs.items():
        if not hasattr(config_object, key):
            raise AttributeError("Config does not have this attribute")
        assert type(value) == type(config_object.key), "Trying to replace a {} attribute by a {}".format(str(type(config_object.key)), str(type(value)))
        config_object.key = value
