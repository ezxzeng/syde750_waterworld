from PIL import Image


def get_frames(env, policy_fn):
    reward_sum = 0
    frame_list = []
    i = 0
    env.reset()

    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        reward_sum += reward
        if done:
            action = None
        else:
            action = policy_fn(observation)

        env.step(action)
        i += 1
        if i % (len(env.possible_agents) + 1) == 0:
            frame_list.append(Image.fromarray(env.render(mode="rgb_array")))

    return reward_sum, frame_list
