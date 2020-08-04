import numpy as np
import torch

class EnergyFunctions(object):
    def __init__(self, chosen_function):
        self.chosen_function = chosen_function

        if self.chosen_function == 1:
            self.probability = self.EF1
        
        elif self.chosen_function == 2:
            self.probability = self.EF2
        
        elif self.chosen_function == 3:
            self.probability = self.EF3
        
        elif self.chosen_function == 4:
            self.probability = self.EF4
    
    def EF1_Old(self, z_input):
        
        
        part1 = 1/2 * ( (torch.norm(z_input, dim = 1) - 2) / 0.4).pow(2)
        part2_1 = torch.exp(-0.5 * ((z_input[:, 0] - 2)/0.6).pow(2)) + torch.exp(-0.5 * ((z_input[:, 0] + 2)/0.6).pow(2))
        
        part2 = torch.log(part2_1)
        Uz = part1 - part2
        
        prob = torch.exp(-1 * Uz)
        
        return prob
    
    def EF1(self, z):
        z1, z2 = z[:, 0], z[:, 1]
        norm = torch.sqrt(z1 ** 2 + z2 ** 2)
        exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
        exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
        u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
        return torch.exp(-u)
    
    def EF2(self, z_input):
        omega1 = -1 * torch.sin(2 * np.pi * z_input[:, 0] / 4)
        Uz = 0.5 * ((z_input[:, 1] - omega1)/ 0.4).pow(2)
        
        prob = torch.exp(-1 * Uz)
        return prob
    
    def EF3(self, z_input):
        
        omega1 = -1 * torch.sin(2 * np.pi * z_input[:, 0] / 4)
        omega2 = 3 * torch.exp(-0.5 * ((z_input[:, 0] - 1)/0.6).pow(2))
        
        part1 = torch.exp(-0.5 * ((z_input[:, 1] - omega1)/0.35).pow(2))
        part2 = torch.exp(-0.5 * ((z_input[:, 1] - omega1 + omega2)/(0.35)).pow(2))
        
        Uz = -1 * torch.log(part1 + part2)
        
        prob = torch.exp(-1 * Uz)
        
        return prob
    
    def EF4(self, z_input):
        
        omega1 = -1 * torch.sin(2 * np.pi * z_input[:, 0] / 4)
        
        sigma = lambda x: 1 / (1 + torch.exp(-1 * x))
        omega3 = 3 * sigma((z_input[:, 0] - 1) / 0.3)
        
        part1 = torch.exp(-0.5 * ((z_input[:, 1] - omega1)/0.4).pow(2))
        part2 = torch.exp(-0.5 * ((z_input[:, 1] - omega1 + omega3)/(0.35)).pow(2))
        
        Uz = -1 * torch.log(part1 + part2)
        
        prob = torch.exp(-1 * Uz)
        
        return prob
        
        