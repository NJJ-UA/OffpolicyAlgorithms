import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class MountainCarImageFeat:
    def __init__(self,  **kwargs):
        self.env = gym.make('MountainCar-v0').unwrapped
        self.env.reset()
        self.num_episodes = kwargs.get('num_episodes', 50)
        self.device = kwargs.get('device')
        self.GAMMA = 0.999
        self.STEP_LIMIT = 5000
        self.last_screen = self.get_screen()
        self.current_screen = self.last_screen
        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        _, _, self.screen_height, self.screen_width = self.last_screen.shape

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n

    def reset(self):
        self.env.reset()
        self.last_screen = self.get_screen()
        self.current_screen = self.last_screen
        return self.get_state()

    def get_state(self):
        return self.current_screen - self.last_screen

    def step(self, action):
        _, reward, is_terminal, info = self.env.step(action.item())
        reward = torch.tensor([reward], device=self.device)
        self.last_screen = self.current_screen
        self.current_screen = self.get_screen()
        return self.get_state(), reward, is_terminal, info


    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])
        return resize(screen).unsqueeze(0).to(self.device)
