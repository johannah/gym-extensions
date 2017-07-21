import gym

def test_import_classic():
    """
    Test import of official envs as seen in the README
    """
    try:
        from gym_extensions.discrete import classic
        envs = [x for x in classic.custom_envs.keys()]
        [gym.make(x) for x in envs]
        return True
    except Exception as inst:
        print("Error:", type(inst))
        print(inst.args)
        return False
   
def test_import_box2d():
    """
    Test import of official envs as seen in the README
    """
    try:
        from gym_extensions.continuous import box2d
        envs = [x for x in box2d.custom_envs.keys()]
        [gym.make(x) for x in envs]
        return True
    except Exception as inst:
        print("Error:", type(inst))
        print(inst.args)
        return False
 
def test_import_gym_navigation_2d():
    """
    Test import of official envs as seen in the README
    """
    try:
        from gym_extensions.continuous import gym_navigation_2d
        envs = [x for x in gym_navigation_2d.custom_envs.keys()]
        [gym.make(x) for x in envs]
        return True
    except Exception as inst:
        print("Error:", type(inst))
        print(inst.args)
        return False

if __name__ == '__main__':
    print(test_import_classic())
    print(test_import_box2d())
    print(test_import_gym_navigation_2d())
