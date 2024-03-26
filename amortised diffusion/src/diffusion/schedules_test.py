from src.diffusion import schedule_discrete, schedule_continuous, sde_diffusion
import torch
import matplotlib.pyplot as plt

lin_dis = schedule_discrete.LinearSchedule()
lin_cont = schedule_continuous.LinearSchedule()

#plot schedule
lin_cont.plot()
lin_dis.plot()

#compare alpha, beta and alpha_bar
lin_cont.alpha(torch.tensor([0.9]))
lin_cont.alpha_bar(torch.tensor([0.9]))
lin_cont.beta(torch.tensor([0.9]))
lin_cont.beta_hardcoded(torch.tensor([0.9]))
1-lin_cont.beta(torch.tensor([0.9]))
1-lin_cont.beta_hardcoded(torch.tensor([0.9]))



hoog_dis = schedule_discrete.HoogeboomSchedule()    
hoog_cont = schedule_continuous.HoogeboomSchedule()

#plot schedule
hoog_cont.plot()    
hoog_dis.plot()

hoog_sde = sde_diffusion.HoogeboomGraphSDE()
vp_sde = sde_diffusion.VPGraphSDE()

def plot_marginal_probabilities(sde,title):
    t = torch.linspace(0, 1, 1000)
    mean,std = sde.marginal_prob(t)
    plt.plot(t, mean, label='$\mu$')
    plt.plot(t, std, label='$\sigma$')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Parameter scaling')
    plt.title(title)
    plt.show()

plot_marginal_probabilities(hoog_sde, "Hoogeboom SDE")
plot_marginal_probabilities(vp_sde, "VP SDE")