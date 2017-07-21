import gym

def test_import_mujoco():
    """
    Test import of official envs as seen in the README
    """
    try:
        from gym_extensions.continuous import mujoco
        envs = [x for x in mujoco.custom_envs.keys()]
        [gym.make(x) for x in envs]
        return True
    except Exception as inst:
        print("Warning MuJoCo Error", type(inst))
        print(inst.args)
        return False
   
if __name__ == '__main__':
    print(test_import_mujoco())
